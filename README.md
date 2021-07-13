# Chef's Hat Cup TRACK 1 - UOT_BUI agent 

## To install the agent libraries:
pip install -r requirements.txt

## To Import:
import uot_biu_agent
agent = uot_biu_agent.UOT_BIU_Agent("name_suffix")   

* The name prefix is UOT_BIU_
* The agent impl IAgent interface (getAction, actionUpdate, observeOthers, getReward)

## To Train:
python run_agent_trainer.py


