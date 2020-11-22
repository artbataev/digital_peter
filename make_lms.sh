ngram-count -text train_texts_for_lm.txt -order 6 -wbdiscount -unk -interpolate -lm lm_train_o6_wbiunk.gz
ngram-count -text train_texts_for_lm.txt -order 6 -wbdiscount -interpolate -lm lm_train_o6_wbi.gz
ngram-count -text train_texts_for_lm.txt -order 5 -wbdiscount -interpolate -lm lm_train_o5_wbi.gz
ngram-count -text train_texts_for_lm.txt -order 6 -wbdiscount -interpolate -maxent -maxent-convert-to-arpa -lm lm_train_o6_wbimaxent.gz
ngram-count -text train_texts_for_lm.txt -order 6 -wbdiscount -lm lm_train_o6_wb.gz
ngram-count -text train_texts_for_lm.txt -order 6 -wbdiscount -gt2min 2 -interpolate -lm lm_train_o6_wbi_gt02.gz



ngram-count -text geval17_for_lm.txt -order 6 -wbdiscount -interpolate -maxent -maxent-convert-to-arpa -lm lm_geval17_o6_wbimaxent.gz
ngram-count -text geval17_for_lm.txt -order 6 -wbdiscount -interpolate -lm lm_geval17_o6_wbi.gz


ngram -order 6 -lm lm_train_o6_wbimaxent.gz -mix-lm lm_geval17_o6_wbimaxent.gz -lambda 0.9 -write-lm lm_train_geval17_06_wbimaxent_0.9.gz
ngram -order 6 -lm lm_train_o6_wbimaxent.gz -mix-lm lm_geval17_o6_wbimaxent.gz -lambda 0.8 -write-lm lm_train_geval17_06_wbimaxent_0.8.gz
ngram -order 6 -lm lm_train_o6_wbimaxent.gz -mix-lm lm_geval17_o6_wbimaxent.gz -lambda 0.7 -write-lm lm_train_geval17_06_wbimaxent_0.7.gz
ngram -order 6 -lm lm_train_o6_wbimaxent.gz -mix-lm lm_geval17_o6_wbimaxent.gz -lambda 0.6 -write-lm lm_train_geval17_06_wbimaxent_0.6.gz