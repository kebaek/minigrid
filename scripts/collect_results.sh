for ((i=1;i<=100;i+=1));
do
  for ((j=3;j<=30;j+=3)); do
      python3 scripts/qlearn.py --env MiniGrid-Maze-v0 --num_episode $j --seed $i
      python3 scripts/qlearn.py --env MiniGrid-Maze-Intermediate-v0 --num_episode $j --seed $i
  done
done

for ((i=1;i<=100;i+=1));
do
  for ((j=3;j<=30;j+=3)); do
      python3 scripts/qlearn_evaluate.py --env MiniGrid-Maze-v0 --num_episode $j --seed $i --dir storage/MiniGrid-Maze-v0/aQL/lr0.10_discount0.90_eps0.80
      python3 scripts/qlearn_evaluate.py --env MiniGrid-Maze-Intermediate-v0 --num_episode $j --seed $i --dir storage/MiniGrid-Maze-Intermediate-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
