for ((i=1;i<=100;i+=1));
do
  for ((j=20;j<=160;j+=20)); do
      python3 scripts/qlearn.py --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir sparse
  done
done

for ((i=1;i<=100;i+=1));
do
  for ((j=20;j<=160;j+=20)); do
      python3 scripts/qlearn_evaluate.py --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir storage/MiniGrid-ThreeDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
