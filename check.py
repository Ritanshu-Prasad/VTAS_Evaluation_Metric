
import json
path = '../Logs and results/New folder/results_new_2/vtas_output/vtas_results.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
    results = data['per_image_results']
    
    print('Average VTAS:', data.get('average_vtas'))
    for r in results:
        cap = r['caption'].lower()
        if 'grass' in cap or 'jacket' in cap or 'woman' in cap or 'skier' in cap:
            print(f'\nCaption: {cap}')
            print('Score:', r.get('vtas_score'))
            print('Tier 1 Matched:', r.get('matched'))
            print('Tier 2 Rescued:', r.get('clip_grounded'))
            print('Hallucinated:', r.get('hallucinated'))
            
except Exception as e:
    print('Error:', e)

