source ~/.virtualenvs/lm_score/bin/activate
workon lm_score
# nohup python3 lm_corrector.py  --lm_name gpt2 --alpha 2 --port 8847 &
python3 lm_corrector.py  --lm_name gpt2-medium --alpha 2 --port 12345
