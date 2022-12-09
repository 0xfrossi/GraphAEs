import xml.etree.ElementTree as ET

class Action:
    def __init__(self, xml_action):
        root = ET.fromstring(xml_action)
        self.name = root.find("Name").text[1:-1]
        self.id = root.find("Id").text
        self.parent = root.find("Parent").text
        self.student = root.find("Student").text
        self.predecessor = [child.find("Id").text for child in root.find("Predecessors")]
        self.precondition = [child.text[1:-1] for child in root.find("Preconds")]
        self.positiveEffects = [child.text[1:-1] for child in root.find("PositiveEffects")]
        self.negativeEffects = [child.text[1:-1] for child in root.find("NegativeEffects")]


def nth_repl(s, sub, repl, nth):
    find = s.find(sub)
    # if find is not p1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != nth:
        # find + 1 means we start at the last match start index + 1
        find = s.find(sub, find + 1)
        i += 1
    # if i  is equal to nth we found nth matches so replace
    if i == nth:
        return s[:find]+repl+s[find + len(sub):]
    return s