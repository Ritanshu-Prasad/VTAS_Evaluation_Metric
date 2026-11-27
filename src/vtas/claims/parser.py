"""Claim parser: caption text -> typed claims, from the spaCy dependency parse.

Claim shapes follow docs/GOLDEN_SET_RUBRIC.md section 2:
  object    "fire hydrant"            noun phrase without article, numeral or adjective
  count     "two dogs"                the caption's own quantity expression
  attribute "car: red"                one claim per adjective, colour, material or state
  relation  "car / under / tree"      preposition between entities, or action with an object
  scene     "beach"                   place type or global condition (scene lexicon)

Count intervals are not decided here: a count claim carries a ``Quantity`` kind and the interval
is read from the policy YAML (``Claim.interval(policy, profile)``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .policy import ConceptPolicy
from .schema import Claim, ParseResult, Quantity

MODELS = {"sm": "en_core_web_sm", "trf": "en_core_web_trf"}
_LEX_PATH = Path(__file__).with_name("lexicon.json")
_NOUN = {"NOUN", "PROPN"}
_PRON_SUBJ = {"he", "she", "it", "they"}
_SKIP_ADJ = {"other", "another", "same", "second", "third", "first", "last", "next", "several", "many",
             "few", "multiple", "numerous", "various", "own", "such", "more", "most", "less", "much"}


def load_lexicon(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or _LEX_PATH).read_text(encoding="utf-8"))


@dataclass
class Ent:
    head: object                        # spaCy token
    name: str
    first: int                          # index of first token of the name span
    plural: bool
    scene: bool = False
    stop: bool = False                  # not a claimable object (image, middle, ...)
    consumed: bool = False              # folded into another entity (container "of", collective "of")
    collective: str | None = None       # collective kind when the head is "group", "crowd", ...
    inner: object = None                # entity head inside "<collective/container> of X"
    quantity: Quantity | None = None
    count_text: str | None = None
    attrs: list = field(default_factory=list)


class ClaimParser:
    def __init__(self, model: str = "trf", policy: ConceptPolicy | None = None, lexicon: dict | None = None,
                 nlp=None):
        self.model = model
        self.policy = policy or ConceptPolicy.load()
        self.lex = lexicon or load_lexicon()
        self.nlp = nlp or self._load(model)
        L = self.lex
        self.multiword = {tuple(p.split()) for p in L["multiword_objects"]}
        self.scene_heads, self.scene_phrases = set(L["scene_heads"]), set(L["scene_phrases"])
        self.stop, self.mass = set(L["stop_nouns"]), set(L["mass_nouns"])
        self.one_item, self.collective_kind = set(L["one_item_plurals"]), L["collective_kind"]
        self.containers, self.positional = set(L["containers"]), set(L["positional_nouns"])
        self.numbers, self.quant_kind = L["numbers"], {k: v for k, v in L["quantifier_kind"].items() if v}
        self.colours, self.materials = set(L["colours"]), set(L["materials"])
        self.idioms = sorted(L["idiom_prepositions"], key=lambda s: -len(s.split()))
        self.posture = set(L["posture_verbs"])
        self.place_rel = set(L["place_relative_nouns"])
        self.adjacency = set(L["adjacency_preps"])
        self.image_nouns = set(L["image_nouns"])
        self.inverse = L["inverse_verbs"]
        self.holders = set(L["holder_nouns"])
        self.aliases = L["scene_aliases"]
        self.rooms = set(L["room_scenes"])
        self.animate = set(L["animate_nouns"])
        self.state_verbs = set(L["state_verbs"])
        self.name_dets = set(L["name_determiners"])
        self.gerund_irr = L["gerund_irregular"]
        self.particle_advs = set(L["particle_advs"])
        self.name_mods = set(L["name_modifiers"]) - {"half", "flat", "spray"}
        self.body_parts = set(L["body_parts"])
        self.clothing = set(L["clothing"])
        self.event_nouns = set(L["event_nouns"])
        self.keep_verb_preps = set(L["keep_verb_preps"])

    @staticmethod
    def _load(model: str):
        import spacy
        name = MODELS.get(model, model)
        try:
            return spacy.load(name)
        except OSError as e:
            raise OSError(f"spaCy model {name!r} is not installed (python -m spacy download {name})") from e

    # ---- public --------------------------------------------------------------------------------

    def parse(self, text: str) -> ParseResult:
        doc = self.nlp(text)
        self._merge_multiword(doc)
        run = _Run(self, doc)
        claims = run.claims()
        return ParseResult(text=text, model=self.model, policy_version=self.policy.version, claims=claims)

    def _merge_multiword(self, doc) -> None:
        spans, i = [], 0
        while i < len(doc) - 1:
            two = (doc[i].lower_, doc[i + 1].lower_)
            if two in self.multiword or (two[0], two[1].rstrip("s")) in self.multiword and two[1].endswith("s"):
                spans.append(doc[i:i + 2])
                i += 2
            else:
                i += 1
        if spans:
            with doc.retokenize() as rt:
                for sp in spans:
                    plural = sp[-1].lower_.endswith("s")
                    rt.merge(sp, attrs={"POS": "NOUN", "TAG": "NNS" if plural else "NN", "DEP": sp.root.dep_,
                                        "LEMMA": sp.text.lower()})


class _Run:
    """One caption."""

    def __init__(self, p: ClaimParser, doc):
        self.p, self.doc, self.L = p, doc, p.lex
        self.ents: dict[int, Ent] = {}
        self.out: list[Claim] = []
        self._seen: set = set()

    # ---- helpers -------------------------------------------------------------------------------

    def add(self, type_, text, tokens=None, **kw) -> None:
        text = " ".join(text.split())
        k = (type_, text.lower())
        if not text or k in self._seen:
            return
        self._seen.add(k)
        s, e = (tokens[0].idx, tokens[-1].idx + len(tokens[-1])) if tokens else (-1, -1)
        self.out.append(Claim(type=type_, text=text.lower(), start=s, end=e, **kw))

    def claims(self) -> list[Claim]:
        self._entities()
        self._object_count_scene()
        self._modifiers()
        self._verbs()
        self._preps()
        self._inverse()
        self._global_scenes()
        return sorted(self.out, key=lambda c: c.start)

    # ---- entities ------------------------------------------------------------------------------

    def _name_tokens(self, h) -> list:
        toks = [h]
        for c in h.lefts:
            if c.lower_ in self.p.name_dets and c.dep_ in ("amod", "det", "advmod"):
                toks = [c] + toks
            if (c.dep_ == "compound" and c.pos_ in _NOUN | {"ADJ"} and c.lower_ not in self.p.colours | self.p.materials
                    and not (c.lower_ == "screen" and h.lower_ in ("tv", "television")))                     or (c.dep_ == "amod" and c.lower_ in self.p.name_mods):
                toks = self._name_tokens(c) + toks
        return toks

    def _entities(self) -> None:
        p = self.p
        for t in self.doc:
            if t.pos_ not in _NOUN or t.dep_ == "poss":
                continue
            if t.dep_ == "compound" and t.head.pos_ in _NOUN and t.head.i > t.i:
                continue
            toks = self._name_tokens(t)
            name = " ".join(x.lower_ for x in toks)
            plural = t.tag_ in ("NNS", "NNPS")
            name = p.aliases.get(name, name)
            e = Ent(head=t, name=name, first=toks[0].i, plural=plural)
            if t.lower_ in ("piece", "pieces", "bit") and any(c.dep_ == "prep" and c.lower_ == "of" and any(
                    g.dep_ == "pobj" for g in c.children) for c in t.children):
                e.collective = "measure"
                e.inner = next(g for c in t.children if c.dep_ == "prep" and c.lower_ == "of"
                               for g in c.children if g.dep_ == "pobj")
                e.stop = True
            elif t.lower_ in p.collective_kind and any(c.dep_ == "prep" and c.lower_ == "of" and
                                                     any(g.dep_ == "pobj" for g in c.children) for c in t.children):
                e.collective = p.collective_kind[t.lower_]
                e.inner = next(g for c in t.children if c.dep_ == "prep" and c.lower_ == "of"
                               for g in c.children if g.dep_ == "pobj")
                e.stop = True
            elif t.lower_ in p.containers and any(c.dep_ == "prep" and c.lower_ == "of" and
                                                  any(g.dep_ == "pobj" for g in c.children) for c in t.children):
                inner = next(g for c in t.children if c.dep_ == "prep" and c.lower_ == "of"
                             for g in c.children if g.dep_ == "pobj")
                if inner.pos_ in _NOUN:
                    e.name = f"{name} of {' '.join(x.lower_ for x in self._name_tokens(inner))}"
                    e.scene = e.name in p.scene_phrases
                    e.inner = inner
                    self.ents.setdefault(inner.i, Ent(head=inner, name="", first=inner.i, plural=False, consumed=True))
            elif name in p.scene_phrases or name.split()[-1] in p.scene_heads:
                e.scene = True
            elif t.lower_ in p.holders and any(c.dep_ == "prep" and c.lower_ == "of" for c in t.children):
                e.inner = next((g for c in t.children if c.dep_ == "prep" and c.lower_ == "of"
                                for g in c.children if g.dep_ == "pobj"), None)
            elif t.lower_ in p.stop or name in p.stop:
                e.stop = True
            self.ents[t.i] = e if not (t.i in self.ents and self.ents[t.i].consumed) else self.ents[t.i]
            if t.i in self.ents and self.ents[t.i].consumed:
                self.ents[t.i].consumed = True
        # entities folded into a container are still referable by relations: keep name
        for i, e in list(self.ents.items()):
            if e.consumed and not e.name:
                toks = self._name_tokens(e.head)
                e.name = " ".join(x.lower_ for x in toks)
                e.plural = e.head.tag_ in ("NNS", "NNPS")

    def quantity_of(self, e: Ent):
        """(Quantity or None, count-claim text or None) for an entity."""
        p, h = self.p, e.head
        qwords = [c for c in h.children if c.dep_ in ("det", "nummod", "amod", "predet", "quantmod", "advmod")]
        for c in qwords:
            n = p.numbers.get(c.lower_)
            if c.dep_ == "nummod" or (n is not None and c.lower_ not in ("one", "both")):
                if c.lower_.isdigit():
                    n = int(c.lower_)
                if n is not None:
                    return Quantity(kind="numeral", n=n, expression=c.lower_), f"{c.lower_} {e.name}"
        for c in qwords:
            k = p.quant_kind.get(c.lower_)
            if k:
                lead = [x for x in c.lefts if x.lower_ == "a"] if c.lower_ == "few" else []
                expr = " ".join(x.lower_ for x in lead) + (" " if lead else "") + c.lower_
                return Quantity(kind=k, expression=expr), f"{expr} {e.name}"
        if e.plural and e.head.lower_ not in p.mass and e.head.lower_ not in p.one_item:
            if any(c.dep_ == "det" and c.lower_ == "the" for c in h.children):
                return None, None  # "the tracks": a known set, no count claim
            return Quantity(kind="plural", expression=""), e.name
        if not e.plural and any(c.dep_ == "det" and c.lower_ in ("a", "an", "the") for c in h.children) \
                or any(c.lower_ == "one" and c.dep_ in ("nummod", "det") for c in h.children):
            return Quantity(kind="singular", expression="a"), None
        return None, None

    def _object_count_scene(self) -> None:
        p = self.p
        for i, e in self.ents.items():
            h = e.head
            if e.consumed:
                continue
            if e.collective == "measure":
                continue
            if e.collective:
                inner = self.ents.get(e.inner.i)
                left = min([c.i for c in h.lefts if c.dep_ in ("det", "amod", "nummod")] + [h.i])
                toks = list(self.doc[left:e.inner.i + 1])
                txt = " ".join(t.lower_ for t in toks)
                kind = e.collective
                if kind == "pair" and (inner is None or e.inner.lower_ in p.one_item):
                    continue
                q = Quantity(kind=kind, expression=" ".join(t.lower_ for t in self.doc[left:h.i + 2]))
                self.add("count", txt, toks, entity=inner.name if inner else e.inner.lower_, quantity=q)
                continue
            if e.stop and not e.scene:
                continue
            if e.scene:
                self.add("scene", e.name, [h], entity=e.name)
                continue
            q, ctext = self.quantity_of(e)
            if h.lower_ in p.collective_kind and not e.plural or h.lower_ in ("crowds",):
                kind = p.collective_kind.get(h.lemma_.lower()) or p.collective_kind.get(h.lower_.rstrip("s"))
                if kind:
                    q = Quantity(kind=kind, expression=h.lower_)
                    lead = [c for c in h.lefts if c.dep_ == "det" and c.lower_ in ("a", "an")]
                    ctext = " ".join(x.lower_ for x in lead + [h])
            e.quantity = q
            if any((g.collective or g.head.lower_ in p.holders) and g.inner is not None and g.inner.i == i
                   for g in self.ents.values()):
                ctext = None  # "a group of X": the collective's own count claim covers X
            self.add("object", e.name, [h], entity=e.name, quantity=q if q and q.kind == "singular" else None)
            if ctext and q.kind != "singular":
                self.add("count", ctext, [h], entity=e.name, quantity=q)
        for i, e in self.ents.items():  # inner nouns of collectives are objects too
            if e.collective and e.inner.i in self.ents:
                inner = self.ents[e.inner.i]
                if not inner.scene and not inner.stop:
                    self.add("object", inner.name, [inner.head], entity=inner.name)

    # ---- attributes ----------------------------------------------------------------------------

    def _attr(self, e: Ent, word, adv: bool = True) -> None:
        if not isinstance(word, str):  # a token: keep adverbs ("dimly lit", "neatly made", "very worn")
            advs = [c.lower_ for c in word.lefts if c.dep_ == "advmod" and c.pos_ == "ADV" and c.lower_ != "not"]
            word = " ".join(advs + [word.lower_]) if adv else word.lower_
        self.add("attribute", f"{e.name}: {word}", [e.head], entity=e.name, attribute=word.lower())

    def _modifiers(self) -> None:
        p = self.p
        for i, e in self.ents.items():
            if e.collective == "measure" and e.inner is not None and e.inner.i in self.ents:
                inner_e, m = self.ents[e.inner.i], e
                for c in m.head.lefts:
                    if c.dep_ == "amod" and c.lower_ not in _SKIP_ADJ:
                        self._attr(inner_e, c)
                continue
            if e.consumed or (e.stop and not e.scene and e.head.lower_ not in p.image_nouns) or not e.name:
                continue
            h = e.head
            done = set()
            for c in h.lefts:
                if c.dep_ == "compound" and c.lower_ in p.colours | p.materials:
                    self._attr(e, c.lower_)
                if c.dep_ == "amod" and c.i not in done and c.lower_ not in _SKIP_ADJ and c.pos_ in ("ADJ", "VERB"):
                    words = [c]
                    for cj in c.conjuncts:
                        if cj.dep_ == "conj" and cj.i > c.i:
                            words.append(cj)
                    done.update(x.i for x in words)
                    if len(words) == 2 and all(w.lower_ in p.colours for w in words):
                        self._attr(e, " and ".join(w.lower_ for w in words))
                    else:
                        for w in words:
                            self._attr(e, w)
            if h.lower_ in ("tv", "television") and any(c.lower_ == "screen" and c.dep_ == "compound" for c in h.lefts):
                adj = [c.lower_ for c in h.lefts if c.dep_ in ("amod",) and c.lower_ not in _SKIP_ADJ]
                self._attr(e, " ".join(adj + ["screen"]))
            for c in h.rights:
                if c.dep_ == "amod" and c.lower_ not in _SKIP_ADJ:
                    self._attr(e, c.lower_)
            # predicate adjective: "the road is winding"
            if h.dep_ in ("nsubj", "nsubjpass") and h.head.lemma_ in p.lex["copulas"]:
                for c in h.head.children:
                    if c.dep_ == "acomp" and c.pos_ == "ADJ":
                        for w in [c, *[x for x in c.conjuncts if x.i > c.i]]:
                            self._attr(e, w)

    # ---- verbs and prepositions ----------------------------------------------------------------

    def subjects_of(self, v) -> list:
        """Entity heads that act as the subject of verb token v."""
        for c in v.children:
            if c.dep_ in ("nsubj", "nsubjpass"):
                if c.lower_ in ("that", "which", "who") and v.dep_ == "relcl":
                    return self._with_conj(v.head)
                return self._with_conj(c) if c.pos_ in _NOUN else []
        if v.dep_ in ("acl", "relcl") and v.head.pos_ in _NOUN:
            return self._with_conj(v.head)
        if v.dep_ in ("conj", "xcomp", "advcl", "ccomp") and v.head.pos_ in ("VERB", "AUX"):
            return self.subjects_of(v.head)
        return []

    def _with_conj(self, t) -> list:
        return [t] + [c for c in t.conjuncts if c.pos_ in _NOUN and c.i != t.i]

    def _ent(self, tok):
        e = self.ents.get(tok.i)
        if e is not None and e.collective and e.inner is not None:
            e = self.ents.get(e.inner.i)
        return e if e and e.name else None

    def _subject_names(self, toks) -> list:
        """Subject name(s): coordinated subjects ("man and woman") are one subject; scene subjects drop."""
        ents, expanded = [], []
        for t in toks:
            r = self.ents.get(t.i)
            if r is not None and r.collective and r.inner is not None:
                expanded += self._with_conj(r.inner)
            else:
                expanded.append(t)
        for t in expanded:
            e = self._ent(t)
            if e is not None and not e.scene and e not in ents \
                    and not (e.stop and e.head.lower_ not in self.p.image_nouns):
                ents.append(e)
        if len(ents) > 1 and all(e.head.pos_ in _NOUN and any(
                x.head.head.i == e.head.i or e.head.head.i == x.head.i for x in ents if x is not e) for e in ents):
            ents = sorted(ents, key=lambda e: e.head.i)
            return [(ents, " and ".join(e.name for e in ents))]
        return [([e], e.name) for e in ents]

    def prep_info(self, prep):
        """(predicate text, [object tokens], positional) for a preposition token, or None."""
        pobjs = [c for c in prep.children if c.dep_ == "pobj"]
        if not pobjs:
            return None
        po = pobjs[0]
        words = [prep]
        if prep.lower_ == "to" and prep.i > 0 and self.doc[prep.i - 1].lower_ in ("next", "close", "adjacent"):
            words = [self.doc[prep.i - 1], prep]
        if po.lower_ in self.p.positional:
            of = next((c for c in po.children if c.dep_ == "prep" and c.lower_ == "of"), None)
            if of is not None:
                inner = [c for c in of.children if c.dep_ == "pobj"]
                if inner:
                    seg = self.doc[prep.i:of.i + 1]
                    return " ".join(t.lower_ for t in seg), self._with_conj(inner[0]), True
            seg = self.doc[prep.i:po.i + 1]
            return " ".join(t.lower_ for t in seg), [], True
        return " ".join(w.lower_ for w in words), self._with_conj(po), False

    def _emit_relation(self, subj_tokens, pred, obj_tokens, positional=False) -> None:
        subjects = self._subject_names(subj_tokens)
        for o in obj_tokens:
            if o.pos_ == "PRON" or (o.lower_ == "other" and any(c.lower_ == "each" for c in o.children)):
                continue  # "it", "each other": no coreference in v1
            oe = self.ents.get(o.i)
            if oe is not None and oe.scene and not positional:
                continue  # "X in/on <place>": the scene claim is enough
            if o.lower_ in self.p.body_parts or o.lower_ in self.p.event_nouns:
                continue  # "in its mouth", "for sale": parts and events are not entities
            if oe is not None and oe.consumed and not oe.name:
                continue
            if oe is not None and oe.collective and oe.inner is not None:
                oe = self.ents.get(oe.inner.i) or oe
            oname = oe.name if oe and oe.name else o.lower_
            for _, sname in subjects:
                self.add("relation", f"{sname} / {pred} / {oname}", [o], subject=sname, predicate=pred,
                         object=oname)

    def _attr_subjects(self, subj_tokens, word: str) -> None:
        for ents, sname in self._subject_names(subj_tokens):
            self.add("attribute", f"{sname}: {word}", [ents[0].head], entity=sname, attribute=word.lower())

    def _place_relative(self, info) -> bool:
        """Positional phrase kept as a relation only for place-relative nouns (middle, center, side...)."""
        return info[2] and any(w in self.p.place_rel for w in info[0].split())

    def _inverse(self) -> None:
        """"plate topped with meat" -> meat / on / plate; "vase of flowers" -> flowers / in / vase."""
        p = self.p
        for v in self.doc:
            if v.pos_ == "VERB" and v.dep_ in ("acl", "amod") and v.head.pos_ in _NOUN:
                for c in v.children:
                    if c.dep_ == "prep":
                        key = f"{v.lower_} {c.lower_}"
                        if key in p.inverse:
                            for o in [x for x in c.children if x.dep_ == "pobj"]:
                                for oo in self._with_conj(o):
                                    self._emit_relation([oo], p.inverse[key], [v.head])
        for t in self.doc:
            if t.lower_ in p.holders and t.pos_ in _NOUN:
                for c in t.children:
                    if c.dep_ == "prep" and c.lower_ == "of":
                        for o in [x for x in c.children if x.dep_ == "pobj"]:
                            for oo in self._with_conj(o):
                                self._emit_relation([oo], "in", [t])

    def _is_animate(self, tok) -> bool:
        e = self._ent(tok)
        names = {tok.lemma_.lower(), tok.lower_}
        if e is not None:
            names |= {e.name, e.head.lemma_.lower(), e.head.lower_}
        return bool(names & self.p.animate)

    def _gerund(self, v) -> str:
        """Attribute verbs are normalised to the -ing form ("walks" -> "walking"); participles stay."""
        if v.tag_ in ("VBG", "VBN"):
            return v.lower_
        lem = v.lemma_.lower()
        if lem in self.p.gerund_irr:
            return self.p.gerund_irr[lem]
        if lem.endswith("e") and not lem.endswith(("ee", "ye", "oe")):
            return lem[:-1] + "ing"
        return lem + "ing"

    def _verbs(self) -> None:
        p = self.p
        for v in self.doc:
            if v.pos_ not in ("VERB", "AUX") or v.dep_ == "amod":
                continue
            copula = v.lemma_ == "be"
            subs = self.subjects_of(v)
            if not subs or v.lemma_ == "have"                     or any(f"{v.lower_} {c.lower_}" in p.inverse for c in v.children if c.dep_ == "prep"):
                continue
            prt = "".join(" " + c.lower_ for c in v.children if c.dep_ == "prt")
            adv = "".join(" " + c.lower_ for c in v.children if c.dep_ == "advmod" and c.lower_ in p.particle_advs)
            vtext = v.lower_ + prt
            posture = v.lemma_ in p.posture
            has_dobj = any(c.dep_ == "dobj" for c in v.children) and not copula
            attr_made = relation_made = False
            for c in v.children:
                if c.dep_ == "dobj" and not copula:
                    if c.lower_ in p.event_nouns or (c.lower_ in ("teeth", "hair") and any(
                            x.dep_ == "poss" for x in c.children)):
                        self._attr_subjects(subs, f"{self._gerund(v)} {c.lower_}")  # "brushing teeth"
                        attr_made = True
                        continue
                    relation_made = True
                    self._emit_relation(subs, vtext, self._with_conj(c))
                elif c.dep_ == "prep":
                    info = self.prep_info(c)
                    if not info:
                        continue
                    objs_scene = all(self.ents.get(o.i) and self.ents[o.i].scene for o in info[1])
                    place_rel = info[2] and self._place_relative(info)
                    if info[2] and not place_rel and objs_scene:
                        continue
                    first = info[0].split()[0]
                    bare = (copula or has_dobj or posture or info[2] or first == "with"
                            or first not in p.keep_verb_preps and first not in ("on", "in", "around", "down", "up"))
                    verb_kept = False
                    if bare:
                        pred = info[0]
                    else:
                        pred, verb_kept = f"{vtext} {info[0]}", True
                    if first == "in" and info[1] and info[1][0].lemma_.lower() in p.clothing and not copula:
                        pred, verb_kept = "wearing", True
                    if verb_kept and not objs_scene:
                        relation_made = True
                    self._emit_relation(subs, pred, info[1], place_rel)
            if copula or v.dep_ == "aux" or v.tag_ not in ("VBG", "VBN", "VBZ", "VBP", "VBD"):
                continue
            word = self._gerund(v) + prt + adv
            if v.lemma_ in p.state_verbs:
                self._attr_subjects(subs, word)       # "parked", "posted": any subject
            elif posture:
                self._attr_subjects([t for t in subs if self._is_animate(t)], word)  # animate subjects only
            elif not relation_made and not attr_made:
                self._attr_subjects(subs, word)

    def _attr_subjects_each(self, subs, word: str) -> None:
        for t in subs:
            self._attr_subjects([t], word)

    def _clause_subject(self, h):
        """Climb pobj -> prep -> head chains to the entity the whole phrase is about."""
        seen = 0
        while seen < 6:
            seen += 1
            prep = h.head if h.dep_ == "pobj" else None
            if prep is None or prep.dep_ != "prep":
                break
            up = prep.head
            if up.pos_ in ("VERB", "AUX"):
                subs = self.subjects_of(up)
                return subs[0] if subs else up
            if up.pos_ not in _NOUN:
                break
            h = up
        return h

    def _preps(self) -> None:
        p = self.p
        for t in self.doc:
            if t.pos_ != "ADP" or t.dep_ != "prep":
                continue
            h = t.head
            idiom = t.lower_ == "to" and t.i > 0 and self.doc[t.i - 1].lower_ in ("next", "close", "adjacent")
            if idiom:
                h = self.doc[t.i - 1].head
            while h.pos_ in ("ADV", "ADJ", "ADP") and h.head.i != h.i:
                h = h.head
            if h.pos_ in ("VERB", "AUX"):
                if idiom:  # "sitting on a table next to a lamp": 'next' hangs on the verb
                    info = self.prep_info(t)
                    subs = self.subjects_of(h)
                    if info and subs:
                        self._emit_relation(subs, info[0], info[1], False)
                continue  # other prepositions are handled with the verb
            if h.pos_ not in _NOUN:
                continue
            info = self.prep_info(t)
            if not info:
                continue
            adjacency = info[0] in p.adjacency or (idiom and info[0] in p.adjacency)
            if adjacency or t.lower_ in ("near", "beside", "by"):
                h = self._clause_subject(h)
            e = self._ent(h)
            raw = self.ents.get(h.i)
            if e is None or (raw is not None and raw.consumed) or (e.stop and e.head.lower_ not in p.image_nouns):
                continue
            if t.lower_ == "of":
                continue
            if e.scene:
                if e.name in p.rooms and info[0] == "with" and not info[2]:
                    names = []
                    for o in info[1]:
                        oe = self.ents.get(o.i)
                        if oe is not None and oe.name and not oe.scene and not oe.stop:
                            names.append(oe.name)
                    if names:
                        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
                        self.add("relation", f"{e.name} / with / {joined}", [t], subject=e.name,
                                 predicate="with", object=joined)
                continue
            if info[2] and not self._place_relative(info):
                if all(self.ents.get(o.i) and self.ents[o.i].scene for o in info[1]):
                    continue
            pred = info[0]
            if pred == "in" and info[1] and info[1][0].lemma_.lower() in p.clothing:
                pred = "wearing"
            self._emit_relation(self._with_conj(h), pred, info[1], self._place_relative(info) if info[2] else False)

    def _global_scenes(self) -> None:
        for m in re.finditer(r"\bat night\b|\bindoors?\b|\boutdoors?\b", self.doc.text.lower()):
            self.add("scene", m.group(0))
