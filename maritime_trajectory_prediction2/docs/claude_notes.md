# 1
Mount this to be in the data directory (we want to use this data)
UbuntuML3GPU% ls -lah /mnt/dataDisk2/aisdata
-rw-rw-r-- 1 johannus johannus 6,2G sep 24 12:51 combined_aiscatcher.log

# 2
Please read through the .md to see which models we have and what objectives there are.

# 3
Right now the repo does not cache the data after it is preprocessed - examine how the preprocessing pipeline works and how we can make this more elegantly. There is no point processing the data more than once unless there has been made changes to the preprocessing logic itself.

Do a write up based on your findings here, then do a task list of the things we need to change.
Afterwards make sure that we work with git best practices and respect the project structure so that we can be a valuable contributer to the project.
Create a new feature branch per task, do the changes, remove irrelevant new artifacts, add proper testing, if everything goes right, merge branch into parent branch and clean up the git environment
