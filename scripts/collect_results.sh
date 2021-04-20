for i in 200 300 400 500 600 700 800 900 1000
do
  python3 -m scripts.qlearn --env MiniGrid-FourDoor-v0 --num_episode $i
  python3 -m scripts.qlearn_evaluate --env MiniGrid-FourDoor-v0 --num_episode $i --dir storage/MiniGrid-FourDoor-v0/aQL/lr0.10_discount0.90_eps0.80
done
