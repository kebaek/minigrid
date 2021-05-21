for i in 100 200
do
  for ((j=1200;j<=2000;j+=200)); do
      python3 ./scripts/qlearn.py --env MiniGrid-FourDoor-v0 --num_episode $j --seed $i --dir 1000
  done
done

for i in 100 200
do
  for ((j=1200;j<=2000;j+=200)); do
      python3 ./scripts/qlearn_evaluate.py --env MiniGrid-FourDoor-v0 --gif "gifs-1000/${j}_${i}" --num_episode $j --seed $i --dir storage/MiniGrid-FourDoor-v0/1000/aQL/lr0.10_discount0.90_eps0.80
  done
done
