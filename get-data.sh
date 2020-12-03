#! /bin/bash

# Hard-coded data location
download_endpoint=e38ee745-6d04-11e5-ba46-22000b92c6ec
download_path=/exalearn-design/neurips/data/output/

# Check if you are logged in with the CLI
globus whoami &> /dev/null
if [ $? != 0 ]; then
    # If not, request user to log in
    globus login 
fi

# Get the id of the local endpoint
if [ $# == 1 ]; then
    # Use a user-supplied
    my_endpoint=$1
else
    # Autodetect
    my_endpoint=$(globus endpoint local-id)
fi

# Check if we can move forward
if [ -n "${my_endpoint}" ]; then
    echo "Detected endpoint: ${my_endpoint}"
else
    echo "Did not find a Globus endpoint on this computer."
    echo
    echo "Either supply one by providing it as an argument (e.g., $0 ebbcf10a-6a44-11fa-958c-0e56c063f437)"
    echo
    echo "Or install one following: https://www.globus.org/globus-connect-personal"
fi

# Determine where to copy the data
data_path=`pwd`/data/output
echo "Writing data to $data_path"

# Start the transfer
globus transfer --label hydronet -r $download_endpoint:$download_path $my_endpoint:$data_path
echo "Transfer has started. Go to http://app.globus.org/activity to track status."
