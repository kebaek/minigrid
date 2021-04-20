for i in 100 200 300 400 500 600 700 800 900 1000
do
  for ((j=10;j<=200;j+=10)); do
    python3 -m scripts.qlearn_evaluate --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir storage/MiniGrid-ThreeDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
  for ((j=300;j<=1000;j+=100)); do
    python3 -m scripts.qlearn_evaluate --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir storage/MiniGrid-ThreeDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
