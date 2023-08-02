Jupyter Notebook提供了一些特殊的命令行魔法函数（Magic Functions），可以帮助你更好地控制和操作Notebook环境，提高工作效率。以下是常用的魔法函数：

1. 行魔法函数（Line Magics）：以`%`为前缀的命令，作用于单行代码。
   - `%run`：运行外部Python脚本。
   - `%load`：将外部脚本导入到代码单元格中。
   - `%time`：计算单行代码的执行时间。
   - `%pwd`：显示当前工作目录。
   - `%who`：列出当前全局变量。
  
2. 单元魔法函数（Cell Magics）：以`%%`为前缀的命令，作用于整个代码单元格。
   - `%%writefile`：将代码单元格内容保存到文件中。
   - `%%time`：计算整个代码单元格的执行时间。
   - `%%html`：在输出区域显示HTML格式的内容。
   - `%%bash`：在命令行中执行Bash命令。
   - `%%capture`：捕获并隐藏代码单元格的输出。

除了上述常用的魔法函数，还有其他许多可用的魔法函数，例如用于调试、性能分析、进度条显示等。要获取完整的魔法函数列表及其说明，可以使用`%lsmagic`命令，它会列出所有可用的魔法函数，并提供简要的说明。

执行以下命令可以显示所有可用的魔法函数：

```
%lsmagic
```

你也可以使用`?`符号来获取单个魔法函数的帮助文档。例如，要查看`%run`命令的帮助信息，可以执行：

```ipython
%run?
```

通过熟练地掌握这些魔法函数，你可以更好地利用Jupyter Notebook进行交互式编程和数据分析。


```
%lsmagic
```


    Available line magics:
    %alias  %alias_magic  %autoawait  %autocall  %automagic  %autosave  %bookmark  %cd  %clear  %cls  
    %colors  %conda  %config  %connect_info  %copy  %ddir  %debug  %dhist  %dirs  %doctest_mode  %echo  
    %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %load  %load_ext  %loadpy  
    %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %macro  %magic  %matplotlib  %mkdir  
    %more  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %pip  %popd  %pprint  
    %precision  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  
    %rehashx  %reload_ext  %ren  %rep  %rerun  %reset  %reset_selective  %rmdir  %run  %save  %sc  
    %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  
    %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%cmd  %%debug  %%file  %%html  %%javascript  %%js  %%latex  
    %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  
    %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



以下是Jupyter Notebook中一些常用的魔法函数：

1. `%run`：运行外部Python脚本。使用格式为`%run script.py`，其中`script.py`是要运行的脚本文件名。

2. `%load`：将外部脚本导入到代码单元格中。使用格式为`%load script.py`，它会将`script.py`中的内容加载到当前代码单元格中。

3. `%time`：计算单行代码的执行时间。使用格式为`%time statement`，其中`statement`是要计时的代码语句。

4. `%pwd`：显示当前工作目录。使用形式为`%pwd`，它会输出当前Notebook所在的目录路径。

5. `%who`：列出当前全局变量。使用形式为`%who`，它会显示当前Notebook中定义的全局变量列表。

6. `%%writefile`：将代码单元格内容保存到文件中。使用形式为`%%writefile filename`，它会将代码单元格的内容保存到`filename`指定的文件中。

7. `%%time`：计算整个代码单元格的执行时间。使用形式为`%%time`，它会计算并显示整个代码单元格的执行时间。

8. `%%html`：在输出区域显示HTML格式的内容。使用形式为`%%html`，后面可以写任意的HTML代码，它会在输出区域渲染显示该HTML内容。

9. `%%bash`：在命令行中执行Bash命令。使用形式为`%%bash`，后面可以写任意的Bash命令，它会在命令行中执行这些命令并显示输出结果。

10. `%%capture`：捕获并隐藏代码单元格的输出。使用形式为`%%capture variable`，它会将代码单元格的输出保存到`variable`指定的变量中，并不会在输出区域显示。



