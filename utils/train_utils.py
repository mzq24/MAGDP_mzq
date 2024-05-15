import pandas as pd
goal_gan_stats = pd.DataFrame({ 'gen_loss': [], 
                                'disc_real_loss': [], 
                                'disc_fake_loss': [], 
                                'disc_loss': [], 
                                'validity_real': [],
                                'validity_fake': []
                                })