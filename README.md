# UOT_BUI agent for:      
##Chef's Hat Cup TRACK 1

# To install the agent libraries:
pip install -r requirements.txt

# To Import Agent:
import uot_biu_agent
agent = uot_biu_agent.UOT_BIU_Agent("name_suffix")   

* the name prefix is UOT_BIU_
* UOT_BIU_Agent impl the IAgent interface (getAction, actionUpdate, observeOthers, getReward)

# To Train the agent
python run_agent_trainer.py


