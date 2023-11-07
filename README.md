# Projects
A bank of projects I found fun and gave up on or I found fun and periodically update here as a backup.


The calculators were some fun projects I made when I first started python and wanted to mimic calculators for expected utility theory and Bayseian analysis. 

Jarvis is a project for developing a robot assistant powered, for now, with Chat-GPT-4. Next is vision/constant environment feedback so it never "turns off". The Jarvis files work with a linux based system. The MacOS updates aren't included here. Continual work in progress.

There are some random functions that will be agents for Jarvis, eventually. Most have not been added to my GitHub.

The twitter bot detects logical fallacies decently, has some bugs but overall works. Not paying $100 a month to use v2 endpoints so I am reworking this to become a web-based application.

Currently working on enhancing the quality of executable tasks that can be performed through prompt engineering via contextual memory and other means (such as taking screen shots and explaining with OCR, executing code on its own in a bash shell, etc.). See ChatBotQuarky for more. Continual work in progress.

The Document Diglet repository is a way to find relevant sections of a document. Basically, using Ctrl-F on transformers with MS MARCO Distillbert v4 from Hugging Face (https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4). In conjunction with the sentence transformers library, this enables us to create sentence embeddings. These embeddings can be queried for with cosine similarity when given a user's prompt. Now that OpenAI's API now has image recognition it's a whole new world.

Simulation World is a project to try out simulations under various conditions. Continual work in progress.

Agent V Agent is an exploration into simulations with LLMs. The plan is to give a 2+ ChatBots a goal in a given environment with rules to abide by and see the choices they make. Eventually this will be scaled up to test for other emergent properties. Continual work in progress.

The RPS Game repository contains a machine learning model that:
1) Is generated from a data frame of 1000 players engaging in a rock paper scissors with a computer over 100 rounds (both are computers but one is called player for simplcities sake).
2) The player and computer both use various strategies to try to out wit one another. The target variable is the computers next move to beat the player. 
3) Once the data is generated we train a MLP model on this and check validation and plot the results.
4) Then we play some games against it and see if it's worthy of playing a human that isn't us. Current iteration is 93% validation without signs of overfitting but playing against it and it will beat a player 50% of the time but does fall victim to spam moves and takes several rounds to adapt.
5) The flask script imports the model and modifies it so it can be used in a SQLite with Flask as an end point hosted on a Heroku server which can be querified through JSON calls on a website through some basic HTML.
6) The true goal is to use the dimensions from the Big 5 was the main features but synthetically generating those distributions based on current studies makes the model over fit. No matter the combination of simplicitly, complexity, model changes, etc it always lead to a perfect accuracy score and tell-tale signs of overfitting. Decided to make a good enough game as a way to still test out the idea but collect the actual data from participants as they play - hence why they enter big 5 personality scores before playing the game in the Flask version.

https://nykkovitali.com/rps to see it in action.
