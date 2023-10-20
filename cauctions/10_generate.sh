nohup python -u _10_generate.py --gpu 5 --eta 0.01 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/001_a150lr1e3/' >./10_generate_001_a150lr1e3.log 2>&1 </dev/null &

nohup python -u _10_generate.py --gpu 5 --eta 0.05 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/005_a150lr1e3/' >./10_generate_005_a150lr1e3.log 2>&1 </dev/null &

nohup python -u _10_generate.py --gpu 5 --eta 0.1 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/010_a150lr1e3/' >./10_generate_010_a150lr1e3.log 2>&1 </dev/null &

nohup python -u _10_generate.py --gpu 5 --eta 0.2 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/020_a150lr1e3/' >./10_generate_020_a150lr1e3.log 2>&1 </dev/null &

nohup python -u _10_generate.py --gpu 5 --eta 0.3 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/030_a150lr1e3/' >./10_generate_030_a150lr1e3.log 2>&1 </dev/null &

nohup python -u _10_generate.py --gpu 5 --eta 0.5 --train_folder './train_files/a150_lr1e3/' --generate_folder './data/generate_primal/050_a150lr1e3/' >./10_generate_050_a150lr1e3.log 2>&1 </dev/null &