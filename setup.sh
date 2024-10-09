pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd eval_agent
pip install -r requirements.txt
cd ..

cd envs/gridworld
pip install -e .
