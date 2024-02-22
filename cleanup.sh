#/bin/bash
# delete the __azurite .json file and the associated folders
rm -rf __blobstorage__/*
rm -rf __queuestorage__/*
rm -rf __tablestorage__/*
rm -rf __filestorage__/*

rmdir __blobstorage__
rmdir __queuestorage__
rmdir __tablestorage__
rmdir __filestorage__

rm __azurite_db_blob__.json
rm __azurite_db_blob_extent__.json
rm __azurite_db_queue__.json
rm __azurite_db_queue_extent__.json
rm __azurite_db_table__.json
