import os
os.chdir("../")

from tests.core.pipelines import merge

def runtime():
    """
    test the times of retrieving something with exact and approximate nearest neighbor
    """
    return NotImplemented

def integration():
    demo = [
        'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
    ]
    merge(demo)

if __name__ == '__main__':
    pass