
import json
path = '../Logs and results/New folder/results_new_2/vtas_output/vtas_results.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
    results = data['per_image_results']
    
    for r in results:
        cap = r['caption'].lower()
        if 'grass' in cap or 'jacket' in cap or 'woman' in cap or 'skier' in cap:
            print(f'\nCaption: {cap}')
            print('CLIP Scores:', r.get('clip_scores'))
            
except Exception as e:
    print('Error:', e)

