for ((j=50; j<=400; j+=50)); do
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo ppo --model ppo-4Door-10-10000-$i --episodes $j --procs 16 --batch-size 300 --max-memory 1500 --seed $i --update-interval 20
done
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.evaluate --env MiniGrid-FourDoor-v0 --episodes 10 --algo ppo --model ppo-4Door-10-10000-$i --procs 10
done

done
