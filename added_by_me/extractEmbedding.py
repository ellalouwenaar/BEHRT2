import torch

def extract_patient_embeddings(model, dataloader, device='cuda'):
    """Extract CLS embeddings voor alle patiënten"""
    model.eval()
    
    patient_embeddings = {}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            age_ids = batch['age_ids'].to(device)
            position_ids = batch['position_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            patient_ids = batch['patient_id']
            
            _, pooled_output = model(
                input_ids, position_ids, segment_ids, age_ids, attention_mask
            )
            
            # Sla embeddings op per patiënt
            for i, patient_id in enumerate(patient_ids):
                patient_embeddings[patient_id] = pooled_output[i].cpu().numpy()
    
    return patient_embeddings