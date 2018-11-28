# Classify Customer's Survey Comments Using Machine Learning in Python

## The Problem

The survey's comments used in the context of this study, are the feedback from customers regarding the IT related services that they received. After their issue is solved, they are prompted to fill out a survey in which they can express their satisfaction.

These comments are then rated in a scale of -1 to 4. 

![alt text](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/Capture.PNG)

For the purpose of this study, the scores -1 and 0 are not considered, since the assigment of these values does not need a special algorithm.

- If it is negative, it will always be -1.
- If the comment is empty, it will be 0.

## The Data

There are XXXX entries in which the text from the comment is available, as well as its score in the format "__label__X" where X can be an integer from 1 to 4. 

| Label        | Text           |
| ------------- |--------------|
| __label__3      | Every single time I have a technical issue, and I contact you all, I receive immediate service!  It surprises me and pleases me every time.  From my experiences, using ITHelpcentral is easy, fast, and worry free.  Thank you! |
| __label__3      | I appreciate the walk-in service very much.  It is invaluable.      |
| __label__2 | amazing !      |

