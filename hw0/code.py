"""
A little code question to understand the formatting and scoring.
Run tests.py to see your score (it will be the last line of output).
Your homework grade may also be based on private tests you can not see.
"""

netid = "lwu4" # replace with your UW UserID!

def hello():
    """
    Return a string of the form "Hello <your netid>".
    For example, mine would return "Hello dshukla".
    """
    return('Hello ' + netid)
    raise(NotImplementedError) # remove this line when you are done

if __name__ == "__main__":
    """
    You can edit code in this if statement for informal testing.
    When you run this file as a python script, it will execute this code.
    This code will not impact the score on the automated test.
    """
    print(hello())
