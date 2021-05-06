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
##### The file system
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
##### Modifying the filesystem
- mkdir: make directory
    - Everything is allowed except for / and something called the null character.
    - Avoid the characters >, <, |, :, &, ;, ?, and *, as they have special meanings in the shell. For fully portable file names, you should stick to characters in the character range [a-zA-Z0-9._-].
- rmdir: remove directory
    - Trying to delete a non-empty directory will result in an error and nothing will happen.
- Copy files: `cp source_files destination`
    - the order of the source files doesn't matter
    - The parameter destination doesn't have to be a directory, it can be a filename. If we wanted to have a copy of the file west in /home/learn called california_love, we could run cp west california_love.
    - `cp /home/dq/prize_winners/.mike /home/dq/prize_winners/mike`
    - should be careful as cp will silently overwrite files with the same name. To protect ourselves against this, enter the interactive mode of cp with the option `-i`.
    - To copy directories, use -R recursively copy everything on subdirectories's subdirectories
    - If `cp -R <folder1> <folder2>`, folder 1 will be copied inside folder2
- rm: delete files
    - To delete directory: rm -R <directory>
    - Have option -i
- mv: move files
    - MUST specify path for both source files and destination if not in home directory, otherwise mistakes.
    - `mv coasts/east coasts/west ./` moves files to the current directory (/ is optional)
    - Does not require -R to move directories
    - Have option -i because files could be overwritten
    - Rename a file or directory: `mv <old_name> <new_name>`

##### Glob patterns and wildcards
-  In the shell, the behavior of * is almost the same: it will match any character, any number of times, except for leading dots (.)
- Passing * as an argument to ls will cause it to list all non-hidden files and directories in the working directory, plus all files at the root of the listed directories.
- `?` matches any character exactly one time
- `\` is an escape character
- We can also escape special characters by using single quotes around the word 
- *square brackets wildcard* matches specific characters: Eg. ls [aiu]* lists items starting with either a, i, or u. [!aiu] matches characters that isn't a, i, or u
- Note about case sensitiveness
- Be mindful of what happens when we use a pattern that doesn't match anything together with ls: Eg. `ls ???`: there aren't any files or directories in the working directory with names that are three characters long, ??? didn't match any filename, and so ??? was passed as an argument to ls without its special meaning.
- Characters ranges such as: [a-z], [A-Z], [0-9], [a-Z]
- Character classes: https://www.gnu.org/software/grep/manual/html_node/Character-Classes-and-Bracket-Expressions.html
    - [:alpha:] usual letters
    - [:digit:] 0-9
    - [:lower:] lowercase letters
    - [:upper:] uppercase letters
    - [:alnum:] letters and numbers
    - Eg: `ls *.[[:lower:]][[:lower:]][[:lower:]]`; `[[:lower:]0]\?` All files starting with either a lowercase letter or zero, followed by a question mark.
- We do not recommend using character ranges, and we should also be careful with character classes:
    - Eg: [a-z] contains a, A, z not Z
    - Locale: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_01
    - POSIX compliant locale: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01
- Should be very careful when using wildcards with commands like rm, cp and mv
    - Before using filesystem-altering commands with wildcards, make sure they'll work as you intend by using them with ls first.
- `find` command: `find [location] -name '['filename']'`. Will search in all subdirectories of the directory. `-name` tells the criteria.
    - To ensure that find behaves as we expect it to with regards to wildcards, we should use single quotes `?a`.
    - `-iname`: ignore case; argument does not need single quotes? 
    - Last time the file was accessed
    - Last time the file was modified
    - What type of file it is (directory, regular file, special file, etc.)
    - Eg. Find all files in the system that end with .b64 and move them into your home directory: 
    ```command line
    find / -name '*.b64'
    mv /sqlite-autoconf-3210000/tea/win/you_found_it.b64 ~
    ```
