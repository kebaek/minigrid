#for ((i=13;i<=23;i+=1));
#do
 # for j in 3150; do
  #    python3 scripts/qlearn.py --env MiniGrid-FourDoor-v0 --num_episode $j --seed $i --dir 1000
   #   python3 scripts/qlearn.py --env MiniGrid-10FourDoor-v0 --num_episode $j --seed $i --dir 1000
  #done
#done

for ((i=13;i<=14;i+=1));
do
  for j in 3150; do
      python3 scripts/qlearn_evaluate.py --env MiniGrid-FourDoor-v0 --num_episode $j  --seed $i --dir storage/MiniGrid-FourDoor-v0/1000/aQL/lr0.10_discount0.90_eps0.80
      python3 scripts/qlearn_evaluate.py --env MiniGrid-10FourDoor-v0 --num_episode $j  --seed $i --dir storage/MiniGrid-10FourDoor-v0/1000/aQL/lr0.10_discount0.90_eps0.80
  done
done
