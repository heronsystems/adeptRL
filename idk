# This script will uninstall all the versions of Python 3.x
# on your Mac OS/X system - It is designed to function as a dry run
# so you can see all that it will do.   To run a dry run, do
#
#   bash uninstall-python3.sh
#
# Examine the output - make sure that it is moving only the expected
# files.  When you are convinced this will do what you expect run
#
#   bash uninstall-python3.sh | sudo bash -v
#
# To verify the files are gone, you can re-run
#
#   bash uninstall-python3.sh
#
# It should produce no output.
#
# And then you can happily re-install Python 3
#
# Written by: Charles R. Severance (@drchuck)
# https://github.com/csev/uninstall-python3
# License: Public Domain / MIT - Use any way you like

ls -l /usr/local/bin | grep /Library/Frameworks/Python.framework/Versions/3 | awk '{print "rm \47/usr/local/bin/" $9 "\47"}'
ls -d /Library/Frameworks/Python.framework/Versions/3.* 2> /dev/null | awk '{print "rm -rf \47" $0 "\47"}'
ls -d /Applications/Python\ 3.* 2> /dev/null | awk '{print "rm -rf \47" $0 "\47"}'
