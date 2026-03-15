#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/11
# @Description: ISP Pipeline


import numpy as np
from path import Path
import os
import yaml
import importlib
import sys
import cv2

class ISP_Pipeline:
    """
    this is a class for ISP Pipeline

    step:
        1. get the ISP Pipeline from yaml
        2. run the ISP Pipeline
        3. get the ISP Pipeline output

    usage:
        isp = ISP_Pipeline(config_path)
    """
    def __init__(self, config_path: str = None, save_intermediate: bool = True) -> None:
        super().__init__()
        self.config_path = config_path
        self.root_path = Path(os.path.abspath(__file__)).parent
        self.__check_envs()
        self.cfg = self.__from_yaml(self.config_path)
        self.pipe = self.__get_isp_pipeline()
        self.input_data = None
        self.step_count = 0  # 用于记录步骤编号
        self.image_id = None
        self.save_intermediate = save_intermediate  # 控制是否保存中间结果

    def __check_envs(self) -> None:
        """
        check the inputs
        """
        assert self.config_path is not None, 'config_path is None, please check it'
        assert os.path.exists(self.config_path), f'config_path {self.config_path} is not exists, please check it'
        sys.path.insert(0, self.root_path + '/algorithm')
        sys.path.insert(0, self.root_path + '/config')
        sys.path.insert(0, self.root_path)
        os.makedirs(self.root_path + '/demo_outputs', exist_ok=True)

    def run(self, input_data: np.ndarray = None, save_intermediate: bool = None) -> np.ndarray:
        if save_intermediate is not None:
            self.save_intermediate = save_intermediate
        if input_data is not None:
            self.input_data = input_data
        return self.__run_isp_pipeline()

    def __from_yaml(self, yaml_path):
        """ Instantiation from a yaml file. """
        if not isinstance(yaml_path, str):
            raise TypeError(
                f'expected a path string but given a {type(yaml_path)}'
            )
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            yml = yaml.safe_load(fp)
        return yml

    def __get_isp_pipeline(self) -> None:
        """
        get ISP Pipeline
        """
        enable_pipeline = self.cfg['enable'].items()
        module = [k for k, v in enable_pipeline if v is True]
        pipe = []
        for m in module:
            py = importlib.import_module(f'algorithm.{m.lower()}')
            cla = getattr(py, m)
            pipe.append(cla)
        return pipe

    def __run_isp_pipeline(self) -> np.ndarray:
        if self.input_data is not None:
            inp = self.input_data
        else:
            from algorithm.fir import FIR
            inp = FIR(**self.cfg).run()
        self.image_id = self.cfg['RAW_img_path'].split('\\')[-1].split('.')[0]
        for p in self.pipe:
            inp = p(inp, **self.cfg).run()
            # 如果不需要保存中间结果，跳过保存步骤
            if self.save_intermediate:
                step_name = p.__name__  # 获取当前模块名称
                self.save_intermediate_result(inp, step_name)
        return inp

    def save_intermediate_result(self, image: np.ndarray, step_name: str) -> None:
        """
        保存中间结果
        
        Args:
            image: 中间结果图像
            step_name: 当前步骤的名称
        """
        # 确保输出目录存在
        output_dir = self.root_path / 'demo_outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一的文件名
        image_id = self.cfg['RAW_img_path'].split('\\')[-1].split('.')[0]
        output_path = output_dir / f'{image_id}_{step_name}_{self.step_count}_output.png'
        
        # 保存图像
        cv2.imwrite(str(output_path), image)
        print(f"中间结果保存成功，路径: {output_path}")
        
        # 增加步骤计数
        self.step_count += 1

    def __save_isp_pipeline_outputs(self, output: np.ndarray) -> None:
        """
        save ISP Pipeline outputs
        """
        import cv2
        image_id = self.cfg['RAW_img_path'].split('\\')[-1].split('.')[0]
        output_path = self.root_path / 'demo_outputs' / f'{image_id}.png'
        cv2.imwrite(str(output_path), output[..., ::-1])
        print(f"最终结果保存成功，路径: {output_path}")