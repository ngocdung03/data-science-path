#### Elements of the Command Line
##### Introduction to the command line
- command-line interfaces (CLIs)
- The terms command-line interface, command language interpreter, shell, console, terminal window and other variants are often used interchangeably.
- Unix shell, namely Bash - common
- A command is case sensitive
- A parameter:
    - Option: a string of symbols that modifies the behavior of the command and it always starts with a dash (-). Can be flag and switch
    - Argument or operand: an object upon which the command acts. The utility (also command or program) is the first item in the instruction.
- Sometimes options are incompatible.
- history:
    - History expansion: very quickly reference lines in the command history by the number to the left of the command
    - Reference commands counting from the end: Eg. run the last: !-1 or !!; next-to-last: !-2
- clear
- exit
- pwd: print working directory. `/`: root (in Unix)
- ls: lists the contents: 
    - -p signals directories from files
    - -A displays hidden files
    - argument /dev: print directory of other location
    - -l display metadata 
    - -h human readable
- cd
    - relative path and it is only valid if working directory is the parent directory
    - can use relative path with every command, eg. ls
    - `cd ~` or `cd .`: to the home directory
    - `cd ~[username]`
    - `cd -` takes us to the penultimate argument we used with cd. This is useful when need to switch back and forth between two directories that do not have a parent-child relationship
    ```cmd
    /home/learn$ cd -
    /home/dq$ cd -
    /home/learn$
    ```