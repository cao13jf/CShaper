for config_file in "./configmemb/test_edt_configs"/*
do
    /bin/python3.6 test_edt.py $config_file
done