cd ~/lab2/lib/python3.11/site-packages/triton/runtime
sed -n 455,470p interpreter.py
sed -n 585,600p interpreter.py
grep -n "class TensorHandle" -A 12 interpreter.py | head -20
