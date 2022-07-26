import os
import sys
import subprocess
import site


class DependencyManager:

    def _init__(self):
        print("dependency manager")

    def install_modules(self, modules):
        """ Adhoc module to install Python dependencies within Python runtime
        is needed to bypass AWS limitation of Glue Shell jobs
        which do not support dependencies"""

        modules = modules.split(",")
        for module in modules:
            # print(f"Installing module {module}...")
            wd = os.getcwd() + "/dependencies"
            # print(wd)
            subprocess.call([
                sys.executable, "-m", "pip", "install", f"--prefix={wd}", module])

            # find nested `site-packages`
            site_packages_dir = self.__find_site_packages_dir(wd)
            if site_packages_dir is None:
                raise Exception(f"`site-packages` is not found inside {wd}")

            site.addsitedir(site_packages_dir)

    def __find_site_packages_dir(self, dr):
        # print(dr)
        for item in next(os.walk(dr))[1]:
            subdir = dr + "/" + item
            if item == 'site-packages':
                return subdir
            ret = self.__find_site_packages_dir(subdir)
            if ret is not None:
                return ret
        return None