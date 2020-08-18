# rlcomp2020 Is A Repo in Order to
record our team's effort in an RL competition held by Fsoft, a company in vietnam.

<b>Ref. URLs</b>
- [https://codelearn.io/game/detail/2212875#ai-game-upload](https://codelearn.io/game/detail/2212875#ai-game-upload)
- [https://github.com/xphongvn/rlcomp2020](https://github.com/xphongvn/rlcomp2020)
- [https://rlcomp.codelearn.io/index.html](https://rlcomp.codelearn.io/index.html)
- [https://www.youtube.com/watch?v=V23ijfCkFI4&feature=youtu.be](https://www.youtube.com/watch?v=V23ijfCkFI4&feature=youtu.be)

## How to Run <code>TestingAgent.py</code> in <code>Miner-Testing-CodeSample/build/</code>?
01. In one terminal, run 
<pre>
# Either not needing to change directory into Miner-Testing-Server/
python Miner-Testing-Server/DUMMY_SERVER.py 1111
# Or changing dir into Miner-Testing-Server/ are both fine.
cd Miner-Testing-Server/
python DUMMY_SERVER.py 1111
</pre>
02. In <b>another</b> terminal, run
<pre>
cd Miner-Testing-CodeSample/build/
python TestingAgent.py
</pre>

## QnA
01. BTC cho mình hỏi maximum số vàng của một ô có giới hạn ntn ạ
    - Map được thiết kế để đảm bảo 4 đội sẽ run trong khoảng 100 step thì vừa hết vàng hoặc gần hết vàng.
Bạn dựa vào đó để tính toán giới hạn vàng nhé. Bạn có thể tham khảo 5 map đấu với bot.
02. On <b><code>gold</code></b> with <code><b>energy=2</b></code>, if the player choose to <code><b>dig</b></code>, then
    - The player will <b>die</b> and this <b>w/o</b> obtaining any gold.
