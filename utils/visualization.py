import torch
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.parametertree import ParameterTree, Parameter, parameterTypes
from pyqtgraph import mkPen, mkColor
from utils.utils import rotate

class Visualization:
    win_title = 'RL4Pedsim'
    win_width, win_height = 1600, 900
    def __init__(self, env, model=None):
        super(Visualization, self).__init__()
        self.app = pg.mkQApp('pedsim')  # 创建 Qt 项目
        self.win = QtWidgets.QMainWindow()  # 创建主窗口
        self.win.setWindowTitle(Visualization.win_title)  # 设置窗口标题
        self.win.resize(Visualization.win_width, Visualization.win_height)  # 设置窗口尺寸
        self.area = DockArea()  # 创建工作区
        self.win.setCentralWidget(self.area)  # 将工作区添加到窗口中间

        self.env = env
        self.model = model
        self.ctrl_dock = None
        self.view_dock = None
        self.ctrl_widget = {}  # ctrl_dock 中的控件
        self.view_widget = {}  # view_widget 中的控件

        self.init_ctrl()
        self.init_view()
    
    def play(self):
        """ 播放 """
        self.win.show()
        self.app.exec_()
        
    def init_ctrl(self):
        """ 初始化 ctrl_widget, 向 ctrl_dock 中添加控件 """
        if self.ctrl_dock is not None:
            self.ctrl_dock.close()
        self.ctrl_dock = Dock('control', size=(Visualization.win_width - Visualization.win_height, Visualization.win_height))  # 创建控制选项卡
        self.area.addDock(self.ctrl_dock, 'left')  # 添加控制选项卡到工作区

        # 计时控件
        self.ctrl_widget['view_timer'] = QtCore.QTimer()  # 设置计时器控件, 用于更新显示结果
        self.ctrl_widget['view_timer'].timeout.connect(self.update)
        self.ctrl_widget['view_timer'].start(50)  # 设置中断间隔 (单位 ms) 并启动
        self.ctrl_widget['act_timer'] = QtCore.QTimer()  # 用于调用 action
        self.ctrl_widget['act_timer'].timeout.connect(self._action)
        self.ctrl_widget['act_timer'].start(int(self.env.meta_data['time_unit'] * 1000))
        self.ctrl_widget['act_timer'].setTimerType(QtCore.Qt.PreciseTimer)

        # info
        w_info = pg.LayoutWidget()
        self.ctrl_dock.addWidget(w_info)
        
        self.ctrl_widget['info'] = pg.DataTreeWidget(data=None)
        w_info.addWidget(self.ctrl_widget['info'])

        # reset & restart
        w0 = pg.LayoutWidget()  # 新建一个 Layout 控件
        self.ctrl_dock.addWidget(w0)  # 将控件添加到 ctrl_dock

        self.ctrl_widget['reset'] = QtWidgets.QPushButton('重设 Reset')  # 生成一个按钮控件
        self.ctrl_widget['reset'].clicked.connect(self._reset)  # 为控件添加回调动作
        w0.addWidget(self.ctrl_widget['reset'])  # 将按钮控件添加到 Layout 控件中

        self.ctrl_widget['restart'] = QtWidgets.QPushButton('重启 Restart')
        self.ctrl_widget['restart'].clicked.connect(self._restart)
        w0.addWidget(self.ctrl_widget['restart'])

        # step & pause & back
        w1 = pg.LayoutWidget()
        self.ctrl_dock.addWidget(w1)

        self.ctrl_widget['back'] = QtWidgets.QPushButton('后退 Back')
        self.ctrl_widget['back'].clicked.connect(self._action_backward)
        w1.addWidget(self.ctrl_widget['back'])

        self.ctrl_widget['pause'] = QtWidgets.QPushButton('暂停/播放 Pause/Play')
        self.ctrl_widget['pause'].clicked.connect(lambda _: self.ctrl_widget['act_timer'].stop() if self.ctrl_widget['act_timer'].isActive() else self.ctrl_widget['act_timer'].start())
        w1.addWidget(self.ctrl_widget['pause'])

        self.ctrl_widget['step'] = QtWidgets.QPushButton('前进 Step')
        self.ctrl_widget['step'].clicked.connect(self._action)
        w1.addWidget(self.ctrl_widget['step'])

        # 其它参数控件
        pw = ParameterTree()
        self.ctrl_dock.addWidget(pw)

        self.ctrl_widget['focus'] = Parameter(name='Focus on: ', type='int', value=-1)  # 当前关注的行人
        pw.addParameters(self.ctrl_widget['focus'])
        
        self.ctrl_widget['speedup'] = Parameter(name='Speed up:', type='float', value=1.0)  # 播放速度
        self.ctrl_widget['speedup'].sigValueChanged.connect(lambda _, v: self.ctrl_widget['act_timer'].setInterval(int(self.env.meta_data['time_unit'] * 1000 / v)))
        pw.addParameters(self.ctrl_widget['speedup'])
        
        self.ctrl_widget['trace_len'] = Parameter(name='Trace Duration: ', type='float', value=1.0)  # 轨迹长度
        pw.addParameters(self.ctrl_widget['trace_len'])
        
        self.ctrl_widget['time_step'] = parameterTypes.SliderParameter(name='Time Step: ', limits=[0, 0])  # 播放进度
        pw.addParameters(self.ctrl_widget['time_step'])

        self.ctrl_widget['direc'] = Parameter(name='移动方向: ', type='bool', value=True)
        pw.addParameters(self.ctrl_widget['direc'])

        self.ctrl_widget['label'] = Parameter(name='行人编号: ', type='bool', value=True)
        pw.addParameters(self.ctrl_widget['label'])

        self.ctrl_widget['trace'] = Parameter(name='显示尾迹: ', type='bool', value=False)
        pw.addParameters(self.ctrl_widget['trace'])

        self.ctrl_widget['des'] = Parameter(name='显示目标: ', type='bool', value=False)
        pw.addParameters(self.ctrl_widget['des'])

        self.ctrl_widget['ray'] = Parameter(name='显示注意力: ', type='bool', value=False)
        pw.addParameters(self.ctrl_widget['ray'])

        self.ctrl_widget['conflict'] = Parameter(name='显示冲突: ', type='bool', value=False and hasattr(self.env, 'CAP_flag'))
        pw.addParameters(self.ctrl_widget['conflict'])

        self.ctrl_widget['predict'] = Parameter(name='显示预测: ', type='bool', value=False)
        pw.addParameters(self.ctrl_widget['predict'])
        

    def init_view(self):
        """ 初始化 view_widget, 向 view_dock 中添加控件. 由于依赖于 ctrl_widget, 因此需要先调用一次 init_ctrl """
        assert len(self.ctrl_widget), "调用 init_view 之前需要先调用 init_ctrl!"
        if self.view_dock is not None:
            self.view_dock.close()
        self.view_dock = Dock('view', size=(Visualization.win_height, Visualization.win_height))  # 创建显示选项卡
        self.area.addDock(self.view_dock, 'right')  # 添加显示选项卡到工作区

        # 添加绘图控件, 并设置背景颜色
        w = pg.PlotWidget(background=(255, 255, 255))
        self.view_dock.addWidget(w)
        xydata = self.env.position[self.env.mask, :]
        xyrange = torch.stack([xydata.min(dim=0).values, xydata.max(dim=0).values], dim=-1) if self.env.mask.sum() > 10 else torch.tensor([[-10., 10.], [-10., 10.]])   # [x/y, min/max]
        w.setRange(xRange=xyrange[0].tolist(), yRange=xyrange[1].tolist())
        w.setAspectLocked()

        # 设置播放进度条
        self.ctrl_widget['time_step'].setValue(0)
        self.ctrl_widget['time_step'].setLimits([0, self.env.num_steps])

        # 目的地箭头
        self.view_widget['arrow'] = pg.ArrowItem(pxMode=False, headLen=0.3, tipAngle=30, baseAngle=60, tailWidth=0.03, brush=None, pen=None)
        w.addItem(self.view_widget['arrow'])

        # 目的地位置点
        self.view_widget['des'] = pg.ScatterPlotItem(pxMode=True, symbol='star', brush='green', pen=None)
        w.addItem(self.view_widget['des'])

        # 行人轨迹
        self.view_widget['trace'] = []
        for _ in range(self.env.num_pedestrians):
            self.view_widget['trace'].append(pg.PlotCurveItem(pen='gray'))
            w.addItem(self.view_widget['trace'][-1])

        # 动作概率
        self.view_widget['logprob'] = pg.ScatterPlotItem(brush=None, size=3, pxMode=True, pen=None)
        w.addItem(self.view_widget['logprob'])


        # 障碍物位置
        self.view_widget['obs'] = pg.ScatterPlotItem(size=self.env.obstacle_radius * 2, pxMode=False, brush=(120, 120, 120), pen='black')
        w.addItem(self.view_widget['obs'])

        # 行人位置
        self.view_widget['ped'] = pg.ScatterPlotItem(size=self.env.ped_radius * 2, pxMode=False, brush=(0, 0, 240), pen=mkPen(cosmetic=False, color='black', width=0.1))
        def test(self_, points, event):
            # print(self_, points, event, points[0]._index)
            self.ctrl_widget['focus'].setValue(points[0]._index if self.ctrl_widget['focus'].value() != points[0]._index else -1)
        self.view_widget['ped'].sigClicked.connect(test)
        w.addItem(self.view_widget['ped'])

        # 冲突图
        self.view_widget['conflict'] = pg.GraphItem(symbolPen=None, symbolBrush=None)
        w.addItem(self.view_widget['conflict'])

        # 冲突预测
        self.view_widget['predict_ttc'] = pg.GraphItem()
        w.addItem(self.view_widget['predict_ttc'])
        self.view_widget['predict_md'] = pg.GraphItem()
        w.addItem(self.view_widget['predict_md'])

        # 注意力状态
        self.view_widget['ray'] = []
        for _ in range(2 * self.env.num_obstacles):
            self.view_widget['ray'].append(pg.ArrowItem(pxMode=False, headLen=0.3, tailWidth=0.03, tipAngle=30, baseAngle=60, brush=None, pen=None))
            w.addItem(self.view_widget['ray'][-1])

        # 行人朝向
        self.view_widget['direc'] = []
        for _ in range(self.env.num_pedestrians):
            self.view_widget['direc'].append(pg.ArrowItem(tipAngle=120, baseAngle=190, headLen=0.3 * self.env.ped_radius, tailLen=None, brush='yellow', pen=mkPen(cosmetic=False, color='black', width=0.01), pxMode=False))
            w.addItem(self.view_widget['direc'][-1])

        # 行人标签
        self.view_widget['label'] = []
        for p in range(self.env.num_pedestrians):
            self.view_widget['label'].append(pg.TextItem(text=str(p), anchor=(0.5, 0.5), color='white'))
            w.addItem(self.view_widget['label'][-1])


    def update(self):
        """ 更新显示 """
        focus = self.ctrl_widget['focus'].value()  # 当前关注的行人
        time_step = self.ctrl_widget['time_step'].value()  # 当前应该显示的时间步
        if time_step >= self.env.num_steps: 
            return
        time_start = max(0, time_step - int(self.ctrl_widget['trace_len'].value() / self.env.meta_data['time_unit']) + 1)  # 历史轨迹的开始时间步

        # 更新行人位置
        if hasattr(self.env, 'CAP_flag') and self.ctrl_widget['conflict'].value():
            brush = np.where(self.env.CAP_flag[:, :, time_step].any(dim=1, keepdim=True), (240, 0, 0), (0, 0, 240))
        else:
            brush = (0, 0, 240)
        self.view_widget['ped'].setData(*self.env.position[:, time_step, :].T, brush=brush)

        # 更新行人标签
        for p in range(self.env.num_pedestrians):
            self.view_widget['label'][p].setPos(*self.env.position[p, time_step, :].numpy())
            self.view_widget['label'][p].setText(str(p) if self.ctrl_widget['label'].value() else '')
        
        # 更新行人朝向
        for idx in range(self.env.num_pedestrians):
            p = self.env.position[idx, time_step, :]
            d = self.env.direction[idx, time_step, 0]
            r = 1.15 * self.env.ped_radius if self.ctrl_widget['direc'].value() else torch.nan
            self.view_widget['direc'][idx].setPos(p[0] + r * d.cos(), p[1] + r * d.sin())
            self.view_widget['direc'][idx].setStyle(angle=180 + d * 180 / torch.pi)

        # 更新障碍物位置
        self.view_widget['obs'].setData(*self.env.obstacle.T if self.env.obstacle is not None else [])

        # 更新目的地位置点
        self.view_widget['des'].setData(*self.env.destination[self.env.mask[:, time_step], :].T if self.ctrl_widget['des'].value() else [])

        # 更新行人轨迹
        for idx in range(self.env.num_pedestrians):
            self.view_widget['trace'][idx].setData(*self.env.position[idx, time_start:time_step + 1, :].T.numpy() if self.ctrl_widget['trace'].value() else [])  # NaN 将自动不被画出

        # 动作概率
        # color = (1- env.logprob[:, time_start:time_step + 1, 0].clamp_(None, 0).exp()).reshape(-1, 1).numpy() * np.array([[0, 255, 255]])
        # self.view_widget['logprob'].setData(*env.position[:, time_start:time_step + 1, :].view(-1, 2).T, brush=color)

        # 目的地箭头
        if 0 <= focus < self.env.num_pedestrians and self.env.mask[focus, time_step] and self.ctrl_widget['des'].value():
            d = self.env.destination[focus, :] - self.env.position[focus, time_step, :]
            r = d.norm(dim=-1)
            a = torch.atan2(d[1], d[0]) / torch.pi * 180
            self.view_widget['arrow'].setPos(*self.env.destination[focus, :])
            self.view_widget['arrow'].setStyle(tailLen=r-0.3, brush='blue', angle=a + 180)
        else:
            self.view_widget['arrow'].setStyle(brush=None)
    
        # 注意力状态
        if 0 <= focus < self.env.num_pedestrians and self.env.mask[focus, time_step] and self.ctrl_widget['ray'].value():
            _, _, s_ext = self.env.get_state(index=time_step)
            p = self.env.position[focus, time_step, :]
            r = s_ext[focus, :, 0]
            a = torch.atan2(s_ext[focus, :, 1], s_ext[focus, :, 2]) + self.env.direction[focus, time_step, 0]
            if self.model is not None:
                weight = self.model.attention.get_weight(self.model.feature(s_ext[focus]))  # (20, 5) -> (20, 1)
                weight01 = torch.where(~weight.isnan(), weight, -torch.inf).softmax(dim=0)
                color = torch.tensor([[255, 255, 0]]) - weight01 * torch.tensor([[0, 255, 0]])
            else:
                color = None
            #     brush = [tuple(c) for c in color.numpy().astype(np.uint8)]
            # else:
            #     brush = 'orange'
            for idx, ray in enumerate(self.view_widget['ray']):
                brush = tuple(color[idx]) if color is not None else 'orange'
                ray.setPos(p[0] + r[idx] * a[idx].cos(), p[1] + r[idx] * a[idx].sin())
                ray.setStyle(tailLen=s_ext[focus, idx, 0] - 0.3, brush=brush, angle=a[idx] * 180 / torch.pi + 180)
        else:
            for idx, ray in enumerate(self.view_widget['ray']):
                ray.setStyle(brush=None)

        # 场景信息
        info = dict(
            meanSpeed = self.env.velocity[self.env.mask[:, time_step], time_step, :].norm(dim=-1).mean().item()
        )

        self.ctrl_widget['info'].setData(info)
        
        # 显示冲突
        if hasattr(self.env, 'CAP_flag') and self.ctrl_widget['conflict'].value():
            msk = self.env.mask[:, time_step]
            pos = self.env.position[msk, time_step, :].numpy()
            adj = self.env.CAP_flag[msk][:, msk][:, :, time_step].nonzero().numpy()
            self.view_widget['conflict'].setData(pos=pos, adj=adj, pen='purple')
        else:
            self.view_widget['conflict'].setData(pen=None)

        # 冲突预测
        if 0 <= focus < self.env.num_pedestrians and self.ctrl_widget['predict'].value():
            _, _, s_ext = self.env.get_state(index=time_step, include_radius=True)
            msk = ~s_ext[focus, :, 0].isnan()
            s_ext = s_ext[focus, msk, :]
            r, s, c, n, t = s_ext.split(1, dim=-1)
            a = torch.atan2(s, c) + self.env.direction[focus, time_step, 0]
            en = torch.cat([c, s], dim=-1)
            et = torch.cat([-s, c], dim=-1)
            x = r * en
            n = n * en
            t = t * et
            v = n + t
            xx = r ** 2
            xv = (x * v).sum(dim=-1, keepdim=True)
            vv = (v ** 2).sum(dim=-1, keepdim=True)
            ttc = (-xv / vv)
            md = (xx - xv ** 2 / vv).sqrt()
            msk = ((ttc > 0) & (ttc < 20) & (md < 2)).squeeze(-1)
            N = msk.sum()
            x, v, ttc, md = x[msk, :], v[msk, :], ttc[msk, :], md[msk, :]

            p0 = self.env.position[(focus,), time_step, :].repeat(N, 1)  # (N, 2)
            p1 = self.env.position[(focus,), time_step, :] + self.env.velocity[(focus,), time_step, :] * ttc  # (N, 2)
            p2 = rotate(x, self.env.direction[(focus,), time_step, 0].repeat(N)) + p0  # (N, 2)
            p3 = rotate(x + v * ttc, self.env.direction[(focus,), time_step, 0].repeat(N)) + p1  # (N, 2)
            pos = torch.cat([p0, p1, p2, p3], dim=0).numpy()
            adj0 = torch.arange(N).view(-1, 1).repeat(1, 2)
            
            adj = torch.cat([adj0 + torch.tensor([0, N]), adj0 + torch.tensor([2*N, 3*N])]).numpy()
            self.view_widget['predict_ttc'].setData(pos=pos, adj=adj, pen='green', brush='green')
                
            adj = (adj0 + torch.tensor([N, 3*N])).numpy()
            self.view_widget['predict_md'].setData(pos=pos, adj=adj, pen='red', brush=None)


    def _reset(self):
        from utils.utils import init_env, get_args
        ARGS = get_args()
        init_env(self.env, ARGS)
        self.ctrl_widget['time_step'].setLimits([0, self.env.num_steps])
        self.ctrl_widget['time_step'].setValue(0)

    def _restart(self):
        self.env.add_pedestrian(self.env.position[:, 0, :], self.env.velocity[:, 0, :], self.env.destination, init=True)
        self.ctrl_widget['time_step'].setLimits([0, self.env.num_steps])
        self.ctrl_widget['time_step'].setValue(0)

    def _action_backward(self):
        time_step = self.ctrl_widget['time_step'].value()
        self.ctrl_widget['time_step'].setValue((time_step - 1) if time_step > 0 else 0)

    def _action(self):
        """ 更新一步 """
        time_step = self.ctrl_widget['time_step'].value()
        if time_step + 1 < self.env.num_steps:  # 有未放完的进度条, 先播放
            self.ctrl_widget['time_step'].setValue(time_step + 1)
        elif 'update' in dir(self.env) and self.env.update():  # env 有 update 方法可以添加时间步
            self.ctrl_widget['time_step'].setValue(time_step + 1)
            self.ctrl_widget['time_step'].setLimits([0, time_step + 1])
        else:  # 不更新时间步
            pass

