import xml.etree.ElementTree as ET

class Plan:
    def __init__(self, plan_description):
        self.plan_name = plan_description
        f = open(plan_description, "r")
        lines = [line for line in f.readlines()]
        lines2 = [l for l in lines if l.find(";;(:metadata") >= 0]
        if len(lines2) == 0:
          print(plan_description)
        init = lines2[0].split(";;(:metadata")[1].split(".")[0][:-2]
        self.states = list()
        for l in lines2[:-1]:
            line = l.rsplit(";;(:metadata",1)[1][:-2]
            line = line.split('<FFheuristic>')[0] + '</Action>'
            action_root = ET.fromstring(line)
            for state_root in action_root.iter('State'):
                state = [child.text[1:-1] for child in state_root]
                self.states.append(state)
        init_root = ET.fromstring(init)
        goal = lines2[len(lines2) - 1].split(";;(:metadata")[1][:-2]
        goals_root = ET.fromstring(goal)
        self.initial_state = [child.text[1:-1] for child in init_root]

        self.goals = [child.text[1:-1] for child in goals_root]