# for ((j=400;j<=2400;j+=200)); do
#     python3 -m scripts.qlearn --env MiniGrid-FourDoor-v0 --num_episode $j
# done

for ((j=400;j<=2400;j+=200)); do
    python3 -m scripts.qlearn --env MiniGrid-ThreeDoor-v0 --num_episode $j
done

for i in 100 200 300 400 500 600 700 800 900 1000
do
  for ((j=400;j<=2400;j+=200)); do
      python3 -m scripts.qlearn_evaluate --env MiniGrid-FourDoor-v0 --num_episode $j --seed $i --dir storage/MiniGrid-FourDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
  for ((j=400;j<=2400;j+=200)); do
      python3 -m scripts.qlearn_evaluate --env MiniGrid-ThreeDoor-v0 --num_episode $j --seed $i --dir storage/MiniGrid-ThreeDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
