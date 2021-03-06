for ((i=1;i<=100;i+=1));
do
 for j in 50 150 350 750 1550 3150; do
     python3 scripts/qlearn.py --env MiniGrid-FourDoor-v0 --num_episode $j --seed $i
     python3 scripts/qlearn.py --env MiniGrid-10FourDoor-v0 --num_episode $j --seed $i
  done
done

for ((i=1;i<=100;i+=1));
do
  for j in 50 150 350 750 1550 3150; do
      python3 scripts/qlearn_evaluate.py --env MiniGrid-FourDoor-v0 --num_episode $j  --seed $i --dir storage/MiniGrid-FourDoor-v0/aQL/lr0.10_discount0.90_eps0.80
      python3 scripts/qlearn_evaluate.py --env MiniGrid-10FourDoor-v0 --num_episode $j  --seed $i --dir storage/MiniGrid-10FourDoor-v0/aQL/lr0.10_discount0.90_eps0.80
  done
done
