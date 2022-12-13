cd test/generated_files/md
for i in {00..10}
do
  cp -r template run_${i}
  cd run_${i}
  sbatch sbatch.sh
  cd ..
done
