"""BotBackServer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView

from BotBackServer import view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', view.check, name='check'),
    path('get_db_dynamic', view.get_db_dynamic, name='get_db_dynamic'),
    path('get_csv_records', view.get_csv_records, name='get_csv_records'),
    path('get_live_positions', view.get_live_positions, name='get_live_positions'),
    path('get_poistion_analyze', view.get_poistion_analyze, name='get_poistion_analyze'),
    path('get_strategy_analyze', view.get_strategy_analyze, name='get_strategy_analyze'),
    path('get_platform_analyze', view.get_platform_analyze, name='get_platform_analyze'),
    path('delete_platform_by_id', view.delete_platform_by_id, name='delete_platform_by_id'),
    path('get_position_bars', view.get_position_bars, name='get_position_bars'),
    path('get_manually_operation', view.get_manually_operation, name='get_manually_operation'),
    path('get_future_balance_analyze', view.get_future_balance_analyze, name='get_future_balance_analyze'),
    path('get_csv_coin_records', view.get_csv_coin_records, name='get_csv_coin_records'),
    path('get_statiscs_hour_date', view.get_statiscs_hour_date, name='get_statiscs_hour_date'),

]
