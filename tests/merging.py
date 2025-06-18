import os
os.chdir("../")

from tests.core import merge

def runtime():
    """test the times of retrieving something with the exact and approximate nearest neighbor"""
    return NotImplemented

def integration():
    demo = [
        'Initialize a logistic regression model. Use standardization on training inputs. Train the model.'
    ]
    merge(demo, save=False)

def validation():
    demo = [
        'Initialize a logistic regression model. Use standardization on training inputs. Train the model.',
        'Create a regression model.',
        'Graph operations'
    ]
    merge(demo)

if __name__ == '__main__':
    validation()