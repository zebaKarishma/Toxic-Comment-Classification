import numpy as np, pandas as pd
lstm_gru_fasttext_glove = pd.read_csv('submission_zk_fgt.csv')#1 - final
bilstm_bigru_fasttext = pd.read_csv('submission_lstm_bigru_ft.csv')#2 - final
bilstm_glove = pd.read_csv('submission_bilstm_glove.csv')#3 - final

logistic_regression = pd.read_csv('submission_lr.csv')#4 

gru_fasttext = pd.read_csv('submission_gru_fasttext_translated.csv')#5 - final
bert = pd.read_csv('bert_submission.csv')#6 - final
lgbm = pd.read_csv('submission_lightGBM.csv')#7 - final

glove_cnn = pd.read_csv('glove_cnn.csv')#8

bi_lstm_gru_cuDNN = pd.read_csv('submission_highacc.csv') #9 -final
text_cnn = pd.read_csv('submission_TextCNN.csv')#10 - final

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
final = bilstm_bigru_fasttext.copy()
final[label_cols] = (lstm_gru_fasttext_glove[label_cols] * 0.07+
     bilstm_bigru_fasttext[label_cols] *  0.07 + 
     bilstm_glove[label_cols]*  0.05 +
     
     gru_fasttext[label_cols]* 0.05+
     text_cnn[label_cols]* 0.05+
     
     bert[label_cols] * 0.33 +
     lgbm[label_cols]* 0.05 + 
     #glove_cnn[label_cols]*0.0625 + 
     #logistic_regression[label_cols] *0.0625 +     
     bi_lstm_gru_cuDNN[label_cols] * 0.33 ) 
final.to_csv('submission_final_08_0.csv', index=False)
print("done !!!")