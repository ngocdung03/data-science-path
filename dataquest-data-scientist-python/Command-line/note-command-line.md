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
##### Users and permissions
- Verify user: `whoami`. Similar to but less verbose than `id -un` 
- Example output of id: `uid=1000(dq) gid=1000(dq) groups=1000(dq),27(sudo)`
    - uid: user ID and name in ()
    - gid: group ID and name of group
    - groups: lists all the groups that the user belongs, with their IDs and their names. If we wish to see what groups our user belongs to, without any aditional information, we can run the command groups. 
- [output-of-list-long.jpg]
- By default the group ownership is given to the primary group of the user who created the file. A primary group is simply a group that is associated with a user. It will typically have the same name and unique number than that of the user.
- In Unix-like system, "everything is a file". The first character of the values in the first column tell us what kind of file it is. `l` tells a symbolic link, similar to a shortcut.
- Permissions have three scopes: for the owner, for the owner group, and for everyone else
    - For each scope, permissions are defined by a sequence of three characters called file access modes. They are, in order, the read mode, the write mode and the execute mode. It's common to call them bits [file-access-mode.jpg]
    - if an ancestor doesn't have execution permissions, none of its descendants will either, regardless of whether the x bit is set or not.
- Change permissions: `chmod permissions files`. `permissions` argument can be divided into three components:
    - Scope: owner/user (u), group (g), others (o), all (a)
    - Operator: add (+), remove (-), set (=)
    - Mode: read(r), write (w), execute(x)
    ```
    # chmod [ugoa][+-=][rwx] files
    chmod u+x my file  # change rw-rw-r-- into rwxrw-r--
    chmod u=rwx,g=rx,o=r my_file  #not including spaces
    ```
    - The operator + causes the selected file mode bits to be added to the existing file mode bits of each file; - causes them to be removed; and = causes them to be added and causes unmentioned bits to be removed.
- stat: see file status
- numeric codification of permissions (octal notation): 
    - all permissions can be obtained from its building blocks -:0, x:1, w:2 and r:4. If we want permission to read and write, we just add the corresponding digits (4 and 2, respectively), giving rw-:6. If we want permission to write and execute, the result is -wx:3 (because 0+2+1=3).
    - The first digit concerns the special permissions
- Root user, administrator, superuser
- `sudo` (superuser do): run commands as if we were other users: `sudo <command>`. `sudo !!`
- Change ownership: `chown` change owner. `chown [new_owner][:new_group] file(s)`. [] means what's inside is optional
- We only had to input the password once. That's because sudo will cache the credentials for 15 minutes (by default).
- From the point of view of the owner of a file, the permissions of the owner have priority over those of its primary group.

#### Text processing in the command line
##### Getting help and reading documentation
- Some programs — specifically those that run with their own executable file — are commands. Eg. Python, Bash
- Programs that are essential to having the system running are usually located in /usr/bin
- 5 different types of commands:
     - file: Usually refered to as program, utility, tool, or sometimes just command, these are simply executable files (analogous to Windows' files whose extension is exe).
    - builtin: Usually refered to as a command or built-in command. Built-in commands are always available in RAM (a special kind of computer memory that is very fast to access, contrary to hard drives). They run quickly and are always available, which is useful when we need to troubleshoot problems in the system.
    - alias: This is just a different name for a command. We usually use aliases to abbreviate frequently used commands.
    - function: A function here is similar to what you learned in Python, only it is in a different language (namely the shell language that we happen to be using).
    - keyword: Keywords are words reserved for specific things and shouldn't be used for other purposes (like being aliases). We'll become more familiar with them when we learn about programming in the shell.
- `type` figure out a command's type: `type pwd`
    - In Bash, probably -t option to get output that is less verbose
    - -P option in Bash
- `declare -F` Commands and features that aren't POSIX compliant and are available in Bash are called bashisms
- `compgen` generate completions for partial names. https://www.gnu.org/software/bash/manual/html_node/Programmable-Completion-Builtins.html
- Set `alias`
```
alias t=type
t pwd t #equivalent to running `type pwd t'
unalias t
```
- When commands of different types have the same name, the priority is in order:
    1. Aliases
    2. Special built-ins (a small list of built-ins that are considered special)
    3. Functions
    4. Regular built-ins
    5. And then it will look for files in a select list of directories called PATH (we'll learn about PATH in the next course)
- Many times, servers will have multiple installations of the same program — possibly with different versions — which can lead to unexpected behavior. By knowing what we're running, we prevent problems from occurring and are able to debug when they happen.
```
python -c "print(3/2)" #1
type python  #python2: / signifies integer division — it ignores the fractional part).
python3 -c "print(3/2)" #1.5
```
- Access documentation: shell <command>
    - Aliases and functions do not have documentation
- Documentation of programs: `man <program>`, use up and down arrow keys to navigate, and "Q" to exit.
        - NAME: The command's name and a brief description of what it does.
        - SYNOPSIS: The allowed syntax.
        - DESCRIPTION: A description of the command. It frequently includes information about its options.
        - OPTIONS: When not included in the section above, the options are documented in this section.
- `whatis` quickly explore the programs in /bin.
- `help <command>`: access documentation for a specific built-in commands in Bash
- help cd: cd [-L|[-P [-e]] [-@]] [dir]
    - cd is mandatory and should be written as is.
    - Every argument is optional.
    - Let's breakdown [-L|[-P [-e]] [-@]].
    - The available options are -L, -P, -e and -@.
    - Since | separates -L from the rest of options, -L can't be used in conjunction with the other options.
    - [-P [-e]] indicates that in case -P is used, we can choose whether or not to use -e, but we can't use -e without -P.
    - We can also elect to use with -@, regardless of whether we include -P.
    - [dir] indicates that we can also include (or not) an optional argument.
    - To constrast with ls, note the lack of ellipsis. This argument isn't repeatable.
    - angle brackets (< >) to signify that an argument is replaceable and mandatory.
- less: Terminal pager. Similar to more. [less-features.jpg]
- Most popular regex used in shells: https://www.gnu.org/software/grep/manual/html_node/Regular-Expressions.html

##### File inspection
- Data: https://github.com/fivethirtyeight/data/blob/master/college-majors/recent-grads.csv
- Non-paging alternatives cd to inspect files: `head -n 5 example_data.csv`. Can use tail
    - An option-argument: an argument that is passed to an option (5 was passed to -n in the example).
    - head -n -5: all except last 5 lines
    - tail -n 15: last 15 lines
    - tail -n +15: all starting from and including line 15
- wc: word count for text file. Output: lines words byte
    - To count characters in the shell's default encoding, we can pass the -m option to wc.
- `column`: prints the contents by columns instead of having it be one long list.
    - `-t`: output like a table
    - `-s`: specify a set of characters for delimiting columns for the -t option. `column -s"," -t example_data.csv`
    - If mistakenly use this command for a very large file, use "Ctrl-C" to interrupt. `less` is a better option for large files.
- `shuf` (shuffle): for extracting random lines from a file.
    - `-n <number>`: display a number of lines
- Most of the files do not have an extension because *nix systems determine what kind of a file is by peeking into its contents and applying some heuristics (like magic numbers).
- `file`: figure out what kind of a file is

##### Text processing
- One of the advantages of the shell over Python is since commands interact more intimately with the filesystem, it tends to be faster for tasks directly concerning input and output of files. It's very common to use the shell to prune a text file to obtain only the information that is relevant to us, and then work on it using Python.
- `cat` concatenates the contents of the arguments, in order, and displays them.
- `tac` concatenates in reverse order of the lines while keeping the order of the files
- `sort` sorts the lines of the files lexicographically.
    - `-r` reverse order
    - `-u` keeps only reverse results
    - with >1 argument: concatenate the files and sort them
    - places each lowercase letter immediately above its uppercase version
    - `-t` the character used for separating the fields
    - `-k` (for key): tell a specific column to sort for:
        - Can pass `-g` together with -k to make the shell sort the numbers numerically.
        - `-r` for reverse order
        - Recerives a range as an argument. Eg 1,1 for only the 1st column. When we pass a range of the form start,stop, sort will look at the columns start through stop as one field only.
    - `sort -t"," -k1,1g example_data_no_header.csv`
- `cut` displays selected columns: 
    - `-d`: tell the specific character as delimiter, equivalent with -t
    - `-f`: specifies the range of fields
    - `cut -d"," -f2,3,7-9 example_data.csv` (doesn't permit reordering)
- `grep` (global regular expressions print.): print lines matching a pattern
    - `-n` displays what line the match corresponds to
    - `-v` get all the lines that do not match the pattern 
    - `-i` the short form of the self-descriptive long option --ignore-case. It makes it so that case does not matter in the pattern
    ```
    # the pattern [aeiou].[aeiou] matches any lowercase vowel, followed by any character (other than new line characters), followed by any lowercase vowel
    grep -n '[aeiou].[aeiou]' characters_no_header
    ```
    - Any lines of characters_no_header that do not end with the number 9: `grep -v '9$' characters_no_header`
    - any lines of any file in rg_data that have fields starting with the word math, while ignoring case: `grep -i ',Math` *`
    - `-E` stands for extended regular expressions. To use the functionality of the listed characters as we learned in Python, we should be using the -E option, while quoting the pattern. In  basic  regular  expressions, the meta-characters ?, +, {, |, (, and ) lose their special meaning; instead use the backslashed versions \?, \+, \{, \|, \(, and \).
    - `-h` exclude filenames
##### Redirection and pipelines
- Output redirection: save output in another file
- echo: print argument to the screen
    - `echo "Trying out >." > my_first_redirection`: save/overwrite the content to the file after redirection operator `>` 
- printf: print formatted
```
# create a file called math_data with contents that are any lines of any file in the directory rg_data that have fields starting with the word math, while ignoring case and excluding the filenames.
grep -hi ',math' rg_data/* > math_data
```
- `>>` appends to the target file, if the target file exists, otherwise it creates a new file. 
- Creating one or more new empty file: `touch` (can overwrite exist files)
- Connecting commands together - pipeline
    - `cut -d"," -f2,5 example_data.csv | grep "^0"`
- To count the number of files we'll pass the output of ls -l /bin to wc -l (the -l option of wc outputs only the number of lines). Since we'll need to exclude the first row (which shows the size of /bin), we'll pipe the output of ls -l /bin to tail -n+2 (this will print all the rows starting from the second one) and then pipe this output to wc -l: `ls -l /bin | tail -n+2 | wc -l`
- count the number of directories.: `ls -l /bin | grep "^d" | wc -l`
- null device, /dev/null is a special file used to discard data. Any data redirected to this file will be ignored by the operating system and simply disappear. This is useful when a command performs an action and outputs something, but we just care about the action.