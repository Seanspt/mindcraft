import os
import sys
sys.path.append('src')
sys.path.append('src/skill_dev_crew')
os.environ["OPENAI_API_BASE"]="https://api.deepseek.com/v1"
os.environ["OPENAI_MODEL_NAME"]="deepseek-chat"
os.environ["OPENAI_API_KEY"]=""

import yaml
from skill_dev_crew.crew import SkillDevelopCrew

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    print("## Welcome to the Skill develop Crew")
    print('-------------------------------')

    with open('./src/skill_dev_crew/config/skilldesign.yaml', 'r', encoding='utf-8') as file:
        skill_design = yaml.safe_load(file)

    with open('../../src/agent/library/skills.js', 'r', encoding='utf-8') as file:
        skill_examples = '\n'.join(file.readlines())

    inputs = {
        'skill' :  skill_design['cut_down_a_tree'],
        'skill_examples': skill_examples
    }

    skill= SkillDevelopCrew().crew().kickoff(inputs=inputs)

    print("\n\n########################")
    print("## Here is the result")
    print("########################\n")
    print("final code for the skill:")
    print(skill)
    

def train():
    """
    Train the crew for a given number of iterations.
    """
    with open('src/skill_dev_crew/config/skilldesign.yaml', 'r', encoding='utf-8') as file:
        skill_design = yaml.safe_load(file)

    with open('../../src/agent/library/skills.js', 'r', encoding='utf-8') as file:
        skill_examples = file.readlines().join('\n')

    inputs = {
        'skill' :  skill_design['cut_down_a_tree'],
        'skill_examples': skill_examples
    }
    try:
        SkillDevelopCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

if __name__ == "__main__":
    run()