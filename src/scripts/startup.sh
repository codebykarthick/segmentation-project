# This needs to be copy pasted in the CMD override before creating the pod
# Copy the ssh key needed and set the correct permission
cp /workspace/ssh_keys/id_ed25519 ~/.ssh/
chmod 0600 ~/.ssh/id_ed25519

# Add the host
ssh-keyscan git.ecdf.ed.ac.uk >> ~/.ssh/known_hosts

# Clone the repo
mkdir -p ~/Developer/Projects/Uni
cd ~/Developer/Projects/Uni
git clone git@git.ecdf.ed.ac.uk:s2681395/cv-mini-project.git

# Cd and run the setup
cd cv-mini-project/src/scripts/
sh cloud.sh