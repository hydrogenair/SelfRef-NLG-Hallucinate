import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_against_human(csv_path: str):
    df = pd.read_csv(csv_path)
    
  
    y_true = df['human'].map({'yes':1, 'no':0, 'Yes':1, 'No':0, '1':1, '0':0}).astype(int)
    y_pred = df['gpt_final'].map({'yes':1, 'no':0, 'Yes':1, 'No':0, '1':1, '0':0}).astype(int)
    
    
    print("\n=== evaluation result (GPT vs Human) ===")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
    
   
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nconfusion matrix:")
    print(pd.DataFrame(cm, 
                      index=['Actual No', 'Actual Yes'],
                      columns=['Predicted No', 'Predicted Yes']))
if __name__ == "__main__":
    output_file = "~/results/Detector/sum/sum_qwen_50.jsonl"
    
   
    evaluate_against_human(output_file)