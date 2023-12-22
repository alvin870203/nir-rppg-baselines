import sys
import subprocess
import signal

train_list = ()
val_list = ()

if len(sys.argv) < 2:
    raise ValueError("No config file specified")

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        exec(open(config_file).read())
        all_subjects = sorted(train_list + val_list)
        print(f"all_subjects = {all_subjects}\n")
    else:
        raise ValueError(f"Unknown argument: {arg}")


for idx in range(len(all_subjects)):
    train_list = tuple(all_subjects[:idx] + all_subjects[idx + 1:])
    val_list = tuple(all_subjects[idx:idx + 1])
    command = f'time python train.py {sys.argv[1]} "--train_list={train_list}" "--val_list={val_list}"'
    try:
        # Execute command using subprocess.run()
        print(command)
        process = subprocess.Popen(command, shell=True)
        # Wait until process terminates
        process.wait()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break
