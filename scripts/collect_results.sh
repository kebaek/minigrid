for ((i=1;j<=100;j+=1));
do
  for ((j=1200;j<=2000;j+=200)); do
      python3 -m scripts.qlearn --env MiniGrid-Maze-v0 --num_episode $j --seed $i
      python3 -m scripts.qlearn --env MiniGrid-Maze-Intermediate-v0 --num_episode $j --seed $i
  done
done

for ((i=1;j<=100;j+=1));
do
  for ((j=1200;j<=2000;j+=200)); do
      python3 -m scripts.qlearn_evaluate --env MiniGrid-Maze-v0 --num_episode $j --seed $i --dir storage/MiniGrid-Maze-v0/aQL/lr0.10_discount0.90_eps0.80
      python3 -m scripts.qlearn_evaluate --env MiniGrid-Maze-Intermediate-v0 --num_episode $j --seed $i --dir storage/MiniGrid-Maze-Intermediate-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
