# classifier/gaussian_classifier_builder.py
import torch
import time
import logging
from .gaussian_classifier import RegularizedGaussianDA, LinearLDAClassifier, LRRGDA
from .base_classifier_builder import BaseClassifierBuilder

def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class LDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.3, device="cuda"):
        self.reg_alpha = reg_alpha
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            lda_reg_alpha=self.reg_alpha
        ).to(self.device)
        
        end_time = time.time()
        log_time_usage("LDA Classifier build", start_time, end_time)
        
        return model


class RegularQDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(
        self,
        qda_reg_alpha1=0.2,
        qda_reg_alpha2=0.2,
        qda_reg_alpha3=0.2,
        device="cuda",
    ):
        self.qda_reg_alpha1 = qda_reg_alpha1
        self.qda_reg_alpha2 = qda_reg_alpha2
        self.qda_reg_alpha3 = qda_reg_alpha3
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        
        # Directly instantiate RegularizedGaussianDA
        model = RegularizedGaussianDA(
            stats_dict=stats_dict,
            class_priors=priors,
            qda_reg_alpha1=self.qda_reg_alpha1,
            qda_reg_alpha2=self.qda_reg_alpha2,
            qda_reg_alpha3=self.qda_reg_alpha3,
        ).to(self.device)
        
        end_time = time.time()
        log_time_usage("Regular QDA Classifier build", start_time, end_time)
        
        return model
    
class LRRGDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(
        self,
        rank=64,
        qda_reg_alpha1=0.2,
        qda_reg_alpha2=0.2,
        qda_reg_alpha3=0.2,
        temperature=1.0,
        device="cuda",
    ):
        self.rank = rank
        self.qda_reg_alpha1 = qda_reg_alpha1
        self.qda_reg_alpha2 = qda_reg_alpha2
        self.qda_reg_alpha3 = qda_reg_alpha3
        self.temperature = temperature
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        
        # Directly instantiate LRRGDA
        model = LRRGDA(
            stats_dict=stats_dict,
            class_priors=priors,
            rank=self.rank,
            qda_reg_alpha1=self.qda_reg_alpha1,
            qda_reg_alpha2=self.qda_reg_alpha2,
            qda_reg_alpha3=self.qda_reg_alpha3,
            temperature=self.temperature,
        ).to(self.device)
        
        end_time = time.time()
        log_time_usage("LR-RGDA Classifier build", start_time, end_time)
        
        return model
