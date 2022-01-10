experiments_path=data/experiments
list_experiments="1tasks_birds 4tasks_birds 10tasks_birds 1tasks_cifar100 4tasks_cifar100 10tasks_cifar100 1tasks_flowers 4tasks_flowers 10tasks_flowers"
for experiment in $list_experiments; do
    results_path=$experiments_path/$experiment
    for file in ${results_path}/*/results; do 
    #echo $(file $pathname); 
    model_name=$(basename $(dirname $file))
    echo $results_path/$model_name
    ls $results_path/$model_name/raw_log-*.txt
    echo "cp $(ls -SF $results_path/$model_name/raw_log-*.txt | head -1) $results_path/$model_name/raw_log.txt"
    cp $(ls -SF $results_path/$model_name/raw_log-*.txt | head -1) $results_path/$model_name/raw_log.txt
    cp $(ls -SF $results_path/$model_name/results/acc_taw-*.txt | head -1) $results_path/$model_name/results/acc_taw.txt
    cp $(ls -SF $results_path/$model_name/results/acc_tag-*.txt | head -1) $results_path/$model_name/results/acc_tag.txt
    cp $(ls -SF $results_path/$model_name/results/forg_taw-*.txt | head -1) $results_path/$model_name/results/forg_taw.txt
    cp $(ls -SF $results_path/$model_name/results/forg_tag-*.txt | head -1) $results_path/$model_name/results/forg_tag.txt
    cp $(ls -SF $results_path/$model_name/results/avg_accs_taw-*.txt | head -1) $results_path/$model_name/results/avg_accs_taw.txt
    cp $(ls -SF $results_path/$model_name/results/avg_accs_tag-*.txt | head -1) $results_path/$model_name/results/avg_accs_tag.txt
    cp $(ls -SF $results_path/$model_name/results/wavg_accs_taw-*.txt | head -1) $results_path/$model_name/results/wavg_accs_taw.txt
    cp $(ls -SF $results_path/$model_name/results/wavg_accs_tag-*.txt | head -1) $results_path/$model_name/results/wavg_accs_tag.txt
    done
done

