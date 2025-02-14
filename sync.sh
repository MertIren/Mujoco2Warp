rsync -av --no-links --info=progress2 -e "ssh -i /home/$USER/.ssh/id_ed25519" ./ParamOpt --exclude 'third_party/*' miren@big-gogh0.lan.local.cmu.edu:/home/miren/Documents/
