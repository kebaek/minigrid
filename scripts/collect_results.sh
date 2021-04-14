for i in 100 200 300 400 500 600 700 800 900 1000
do
  python3 -m scripts.qlearn --env MiniGrid-ThreeDoor-v0 --num_episode $i
done

for i in 100 200 300 400 500 600 700 800 900 1000
do
  python3 -m scripts.qlearn_evaluate --env MiniGrid-ThreeDoor-v0 --num_episode $i --dir storage/MiniGrid-ThreeDoor-v0/aQL/lr0.10_discount0.90_eps0.80
done
