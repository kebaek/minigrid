for ((i=1;i<=100;i+=1));
do
  for ((j=20;j<=160;j+=20)); do
      python3 scripts/qlearn.py --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir sparse
  done
