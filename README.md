# Chef's Hat Cup TRACK 1 - UOT_BUI agent 

## To install the agent libraries:
pip install -r requirements.txt

## To Import:
import uot_biu_agent
agent = uot_biu_agent.UOT_BIU_Agent("name_suffix")   
<br>
* The name prefix is UOT_BIU_
* The agent impl IAgent interface (getAction, actionUpdate, observeOthers, getReward)
* The agent expects to find the policy.pth file in the running folder. <br>
  if this is not the case, you can use the policy_file_path arg to pass the policy full path.<br>
  for example:<br>
  agent = uot_biu_agent.UOT_BIU_Agent("name_suffix", policy_file_path="/mnt/tmp/policy.pth")<br>
## To Train:
python run_agent_trainer.py

