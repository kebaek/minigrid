for i in 100 200 300 400 500 600 700 800 900 1000
do
  for j in 10 20 30 40 50 60 70 80 90 100
  do
    python3 -m scripts.qlearn_evaluate --env MiniGrid-Maze-v0 --num_episode $j --seed $i --dir storage/MiniGrid-Maze-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
